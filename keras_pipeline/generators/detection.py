""" Implemented with reference from NumpyArrayIterator at
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""

import math
import numpy as np
import random
import threading

import keras

from ._image_generator import ImageGenerator

from ..utils.anchors import (
    anchor_targets_bbox,
    bbox_transform,
    compute_all_anchors
)

from ..preprocessing.image_transform import (
    adjust_transform_for_image,
    apply_transform,
    resize_image_1
)

from ..preprocessing.transform import transform_aabb


class DetectionGenerator(ImageGenerator):
    def __init__(self, config):
        # Store general dataset info
        self.dataset         = config.dataset
        self.all_image_index = config.dataset.list_all_image_index()
        self.size            = config.dataset.get_size()
        self.num_classes     = config.dataset.get_num_classes()
        self.label_to_name   = config.dataset.label_to_name

        # Typical generator config
        self.batch_size     = config.batch_size
        self.image_min_side = config.image_min_side
        self.image_max_side = config.image_max_side
        self.group_method   = config.group_method
        self.shuffle        = config.shuffle_groups

        # Create transform generator
        self.transform_parameters = config.transform_parameters
        self.transform_generator = None
        if config.allow_transform:
            self.transform_generator = self._make_transform_generator(config)

        # Compute anchors variables
        self.anchor_sizes   = config.anchor_sizes
        self.anchor_strides = config.anchor_strides
        self.anchor_ratios  = config.anchor_ratios
        self.anchor_scales  = config.anchor_scales
        self.compute_pyramid_feature_shapes_for_img_shape = \
            config.compute_pyramid_feature_shapes_for_img_shape

        # Validate dataset
        self._validate_dataset()

        # Tools which helps order the data generated
        self.lock = threading.Lock() # this is to allow for parrallel batch processing
        self.group_index_generator = self._make_index_generator()


    ###########################################################################
    #### This marks the start of helper functions

    def _validate_dataset(self):
        """ Dataset validator which validates the suitability of the dataset """
        img_id = self.dataset.list_all_image_index()[0]

        size         = self.dataset.get_size()
        num_classes  = self.dataset.get_num_classes()
        aspect_ratio = self.dataset.get_image_aspect_ratio(img_id)
        assert isinstance(size, int), 'Size must be an integer, got {}'.format(type(size))
        assert isinstance(num_classes, int), 'Num classes must be an integer, got {}'.format(type(num_class))
        assert isinstance(aspect_ratio, float), 'Aspect ratio must be a float, got {}'.format(type(aspect_ratio))

        img = self.dataset.load_image(img_id)
        assert (len(img.shape) == 3) and (img.shape[-1] == 3), 'img is of wrong shape, got {}'.format(img.shape)

        ann = self.dataset.get_annotations_array(img_id)
        assert ann.shape[1] == 5, 'get_annotations_array should return a (None, 5) shaped array, got {}'.format(ann.shape)

        label = self.dataset.label_to_name(num_classes - 1)

    def _group_image_ids(self):
        """ Group img_ids according to batch_size and group_method """
        # Retrieve all image ids and count number of groups
        img_ids = self.all_image_index
        num_groups = math.ceil(len(img_ids) / self.batch_size)

        # Perform grouping
        if self.group_method == 'random':
            random.shuffle(img_ids)
        elif self.group_method == 'ratio':
            img_ids.sort(key=lambda x: self.dataset.get_image_aspect_ratio(x))

        # Group image ids
        groups = []
        for group_i in range(num_groups):
            start = self.batch_size * group_i
            end   = self.batch_size * (group_i + 1)
            groups.append(img_ids[start:end])

        # Perform shuffing
        if self.shuffle:
            random.shuffle(groups)

        # Save groups
        self.groups = groups

    def compute_anchors(self, image_shape):
        return compute_all_anchors(
            image_shape,
            sizes = self.anchor_sizes,
            strides = self.anchor_strides,
            ratios = self.anchor_ratios,
            scales = self.anchor_scales,
            shapes_callback = self.compute_pyramid_feature_shapes_for_img_shape,
        )


    ###########################################################################
    #### This marks the start of _get_batches_of_transformed_samples helper functions

    def load_X_group(self, group):
        """ Loads the raw group images from the dataset """
        return [self.dataset.load_image(image_index) for image_index in group]

    def load_Y_group(self, group):
        """ Loads the raw group annotations from the dataset
        Annotations are of the shape (None, 5),
        each detection is in the format (x1, y1, x2, y2, class)
        """
        return [self.dataset.get_annotations_array(image_index) for image_index in group]

    def filter_annotations(self, image, annotations):
        assert isinstance(annotations, np.ndarray)

        # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
        invalid_indices = np.where(
            (annotations[:, 2] <= annotations[:, 0]) |
            (annotations[:, 3] <= annotations[:, 1]) |
            (annotations[:, 0] < 0) |
            (annotations[:, 1] < 0) |
            (annotations[:, 2] > image.shape[1]) |
            (annotations[:, 3] > image.shape[0])
        )[0]

        if len(invalid_indices):
            annotations = np.delete(annotations, invalid_indices, axis=0)

        return image, annotations

    def random_transform_entry(self, image, annotations):
        transformation = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

        # Transform image and annotations
        image = apply_transform(transformation, image, self.transform_parameters)
        annotations = annotations.copy()
        for index in range(annotations.shape[0]):
            annotations[index, :4] = transform_aabb(transformation, annotations[index, :4])

        return image, annotations

    def resize_image(self, image):
        return resize_image_1(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_entry(self, image, annotations):
        """ Preprocesses an entry """
        # Filter invalid annotations
        image, annotations = self.filter_annotations(image, annotations)

        # Apply transformation
        if self.transform_generator:
            image, annotations = self.random_transform_entry(image, annotations)

        # resize image and annotations
        image, image_scale = self.resize_image(image)
        annotations[:, :4] *= image_scale

        return image, annotations

    def compute_inputs(self, image_group):
        # Get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # Construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # Copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets(self, image_group, annotations_group):
        # Get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # Compute labels and regression targets
        labels_group     = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            labels_group[index], annotations, anchors = anchor_targets_bbox(
                max_shape,
                annotations,
                self.num_classes,
                mask_shape = image.shape,
                compute_anchors = self.compute_anchors
            )
            regression_group[index] = bbox_transform(anchors, annotations)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states           = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch     = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression

        return [labels_batch, regression_batch]
