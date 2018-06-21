""" Implemented with reference from NumpyArrayIterator at
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""

import math
import numpy as np
import random
import threading
import warnings

import keras

from ..utils.anchors import (
    anchor_targets_bbox,
    bbox_transform,
    compute_all_anchors
)

from ..preprocessing.image_transform import (
    adjust_transform_for_image,
    apply_transform,
    resize_image
)

from ..preprocessing.transform import (
    random_transform_generator,
    transform_aabb
)


""" These functions written outside are those that
    1. does not reference self and
    2. won't be called (or have the potential to be called) after __init__ is done
"""

def _group_image_ids(img_ids, batch_size, group_method, dataset):
    """ Group img_ids according to batch_size and group_method """
    # Make copy of ids, we're going to order them later
    img_ids = img_ids[:]

    if group_method == 'random':
        random.shuffle(img_ids)
    elif group_method == 'ratio':
        img_ids.sort(key=lambda x: dataset.get_image_aspect_ratio(x))

    num_groups = math.ceil(len(img_ids) / batch_size)

    groups = []
    for group_i in range(num_groups):
        start = batch_size * group_i
        end   = batch_size * (group_i + 1)
        groups.append(img_ids[start:end])

    return groups


def _make_transform_generator(config):
    return random_transform_generator(
        min_rotation    = config.min_rotation,
        max_rotation    = config.max_rotation,
        min_translation = config.min_translation,
        max_translation = config.max_translation,
        min_shear       = config.min_shear,
        max_shear       = config.max_shear,
        min_scaling     = config.min_scaling,
        max_scaling     = config.max_scaling,
        flip_x_chance   = config.flip_x_chance,
        flip_y_chance   = config.flip_y_chance,
    )

def _validate_dataset(dataset):
    """ Validates that dataset is suitable for a detection task
    This function doesn't really generate a proper Error functions but the error log should be enough for debugging purposes
    """
    img_id = dataset.list_image_index()[0]

    dataset.get_size()
    dataset.get_num_object_classes()
    dataset.get_image_aspect_ratio(img_id)

    dataset.load_image(img_id)
    dataset.load_image_info(img_id)

    dataset.load_annotations(img_id)
    dataset.load_annotations_array(img_id)



class DetectionGenerator(object):
    def __init__(self, config):
        _validate_dataset(config.dataset)

        # here dataset is the only object attribute
        # generally object and callable attributes in the generator is
        # ill adviced due to to their sometimes unpickable nature
        # The DetectionDataset object is picklable
        self.data            = config.dataset
        self.size            = config.dataset.get_size()
        self.num_classes     = config.dataset.get_num_object_classes()
        self.label_to_name   = config.dataset.object_class_id_to_object_class

        # Typical generator config
        self.batch_size      = config.batch_size
        self.image_min_side  = config.image_min_side
        self.image_max_side  = config.image_max_side
        self.shuffle         = config.shuffle_groups

        # Compute anchors variables
        self.anchor_sizes   = config.anchor_sizes
        self.anchor_strides = config.anchor_strides
        self.anchor_ratios  = config.anchor_ratios
        self.anchor_scales  = config.anchor_scales
        self.compute_pyramid_feature_shapes_for_img_shape = \
            config.compute_pyramid_feature_shapes_for_img_shape

        # Define groups (1 group ==  1 batch)
        self.groups = _group_image_ids(
            config.dataset.list_image_index(),
            config.batch_size,
            config.group_method,
            config.dataset
        )

        # Create transform generator
        self.transform_parameters = config.transform_parameters
        self.transform_generator = None
        if config.allow_transform:
            self.transform_generator = _make_transform_generator(config)

        # Create the tools which helps define the order in which the
        self.lock = threading.Lock() # this is to allow for parrallel batch processing
        self.group_index_generator = self._make_index_generator(len(self.groups))

    ###########################################################################
    #### Detection Specific functions

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
    #### This marks the start of _get_batches_of_transformed_samples helpers

    def load_image_group(self, group):
        """ Returns list of images from group """
        return [self.data.load_image(image_index) for image_index in group]

    def load_annotations_group(self, group):
        """ Returns list of annotations from group
        Annotations should be in the form [x1, y1, x2, y2, class]
        Notably this is different to keras_pipeline.datasets.ImageDatasetTemplate.load_annotations_array
        """
        annotations_group = []
        for image_index in group:
            annotations = self.data.load_annotations_array(image_index)

            # for annotations_group bbox must be in x1, y1, x2, y2 format
            annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
            annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

            annotations_group.append(annotations)

        return annotations_group

    def filter_annotations(self, image_group, annotations_group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
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

            # delete invalid indicies
            if len(invalid_indices):
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def random_transform_entry(self, image, annotations):
        transformation = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

        # Transform image and annotations
        image = apply_transform(transformation, image, self.transform_parameters)
        annotations = annotations.copy()
        for index in range(annotations.shape[0]):
            annotations[index, :4] = transform_aabb(transformation, annotations[index, :4])

        return image, annotations

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_entry(self, image, annotations):
        # Apply transformation
        if self.transform_generator:
            image, annotations = self.random_transform_entry(image, annotations)

        # resize image and annotations
        image, image_scale = self.resize_image(image)
        annotations[:, :4] *= image_scale

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # Preprocess a single group entry
            image, annotations = self.preprocess_entry(image, annotations)

            # Update group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

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
                self.data.num_classes,
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

    ###########################################################################
    #### This marks the end of _get_batches_of_transformed_samples helpers

    def _get_batches_of_transformed_samples(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group)

        # perform preprocessing
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compuate network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def _make_index_generator(self, num_groups):
        """ Returns a generator which yields group index to train the model in """
        # start group_index at -1 so that first group_index returned is 0
        group_index = -1
        reset_point = num_groups - 1

        while True:
            if group_index < reset_point:
                group_index += 1
            else:
                random.shuffle(self.groups)
                group_index = 0
            yield group_index

    def next(self):
        with self.lock:
            group_index = next(self.group_index_generator)
            group = self.groups[group_index]
        return self._get_batches_of_transformed_samples(group)

    def __len__(self):
        return self.data.size

    def __next__(self):
        return self.next()

    def __iter__(self):
        warnings.warn('Iteration of Generator is still in an experimental phase')
        group_index = -1
        reset_point = len(self.groups) - 1

        while group_index < reset_point:
            group_index += 1
            group = self.groups[group_index]
            yield self._get_batches_of_transformed_samples(group)

    ###########################################################################
    #### This marks the start of all evaluation only methods

    def create_eval_generator(self):
        """ Creates an optimized evaluation set generator
        At present evaluation is working only at batch sizes of 1

        Generator would return original images and annotations along with network inputs and targets.
        All images and annotations are scaled down to previously defined size ranges.

        Generator generates items in the following format
        [network_inputs, orig_image], [classification_targets, regression_targets, orig_annotations, image_scale]
        """

        img_ids = self.data.list_image_index()
        num_img = len(img_ids)
        i = 0

        while i < num_img:
            img_id = img_ids[i]

            # load images and annotations
            image_group       = self.load_image_group([img_id])
            annotations_group = self.load_annotations_group([img_id])

            # Check validity of annotations
            image_group, annotations_group = self.filter_annotations(image_group, annotations_group)

            # Get original image and annotations
            orig_image = image_group[0]
            orig_annotations = annotations_group[0]

            # Perform resizing
            scaled_image, image_scale = self.resize_image(orig_image)
            scaled_annotations = orig_annotations.copy()
            scaled_annotations[:, :4] *= image_scale

            # Recompile into group from
            image_group = [scaled_image]
            annotations_group = [scaled_annotations]

            # Compuate network inputs
            inputs = self.compute_inputs(image_group)

            # Increase counter
            i += 1

            yield [inputs, orig_image], [orig_annotations, image_scale]
