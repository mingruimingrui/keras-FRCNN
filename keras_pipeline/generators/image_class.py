""" Implemented with reference from NumpyArrayIterator at
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""

import math
import numpy as np
import random
import threading
import warnings

import keras

from ..preprocessing.image_transform import (
    adjust_transform_for_image,
    apply_transform,
    resize_image_2
)

from ..preprocessing.transform import random_transform_generator

""" These functions written outside are those that
    1. does not reference self and
    2. won't be called (or have the potential to be called) after __init__ is done
"""

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
    dataset.get_num_image_classes()
    dataset.get_image_aspect_ratio(img_id)

    dataset.load_image(img_id)
    dataset.load_image_info(img_id)
    dataset.load_image_bbox_array(img_id)
    dataset.load_image_class_array(img_id)


class ImageClassGenerator(object):
    def __init__(self, config):
        _validate_dataset(config.dataset)

        # here dataset is the only object attribute
        # generally object and callable attributes in the generator is
        # ill adviced due to to their sometimes unpickable nature
        # The DetectionDataset object is picklable
        self.data            = config.dataset
        self.size            = config.dataset.get_size()
        self.num_classes     = config.dataset.get_num_image_classes()
        self.label_to_name   = config.dataset.image_class_id_to_image_class

        # Typical generator config
        self.batch_size      = config.batch_size
        self.image_height    = config.image_height
        self.image_width     = config.image_width
        self.stretch_to_fill = config.stretch_to_fill
        self.shuffle         = config.shuffle

        # Store image id order
        self.groups = self._group_image_ids()

        # Create transform generator
        self.transform_parameters = config.transform_parameters
        self.transform_generator = None
        if config.allow_transform:
            self.transform_generator = _make_transform_generator(config)

        # Create tools for multi threading
        self.lock = threading.Lock() # this is to allow for parrallel batch processing
        self.group_index_generator = self._make_index_generator()

    ###########################################################################
    #### This marks the start of _get_batches_of_transformed_samples helpers

    def load_image_group(self, group):
        """ Returns list of images from group """
        return [self.data.load_image(image_index) for image_index in group]

    def load_bbox_group(self, group):
        """ Returns list of bboxes from group """
        return [self.data.load_image_bbox_array(image_index) for image_index in group]

    def load_labels_group(self, group):
        """ Returns list of labels
        labels will be in the form [classes] """
        return [self.data.load_image_class_array(image_index) for image_index in group]

    def random_transform_image(self, image):
        transformation = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

        # Transform image and annotations
        image = apply_transform(transformation, image, self.transform_parameters)

        return image

    def resize_image(self, image):
        return resize_image_2(
            image,
            width=self.image_width,
            height=self.image_height,
            stretch_to_fill=self.stretch_to_fill
        )

    def preprocess_entry(self, image, bbox, returns_unscaled=False):
        # Crop image
        if bbox.shape == (4,):
            # Transform bbox into the form [x1, y1, x2, y2, class]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(b) for b in bbox]

            # Get cropped image
            image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Apply transformation
        if self.transform_generator:
            image = self.random_transform_image(image)

        # resize image
        image = self.resize_image(image)

        return image

    def preprocess_image_group(self, image_group, bbox_group):
        for index, (image, bbox) in enumerate(zip(image_group, bbox_group)):
            # Preprocess a single group entry
            image = self.preprocess_entry(image, bbox)

            # Update group
            image_group[index]  = image

        return image_group

    def compute_inputs(self, image_group):
        # Construct an image batch object
        image_batch = np.zeros((self.batch_size, self.image_height, self.image_width, 3), dtype=keras.backend.floatx())

        # Copy all images to the center of the image batch object
        for image_index, image in enumerate(image_group):
            x1 = math.floor((self.image_width  - image.shape[1]) / 2)
            y1 = math.floor((self.image_height - image.shape[0]) / 2)
            x2 = x1 + image.shape[1]
            y2 = y1 + image.shape[0]
            image_batch[image_index, y1:y2, x1:x2] = image

        return image_batch

    def compute_targets(self, labels_group):
        # Construct a labels batch object
        labels_batch = np.zeros((self.batch_size, self.num_classes), dtype=keras.backend.floatx())

        # Copy all label info to the batch object
        for labels_index, labels in enumerate(labels_group):
            labels_batch[labels_index, labels] = 1

        return labels_batch

    ###########################################################################
    #### This marks the end of _get_batches_of_transformed_samples helpers

    def _get_batches_of_transformed_samples(self, group):
        # load images bboxes, and labels
        image_group  = self.load_image_group(group)
        bbox_group   = self.load_bbox_group(group)
        labels_group = self.load_labels_group(group)

        # perform preprocessing on image
        image_group = self.preprocess_image_group(image_group, bbox_group)

        # compuate network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(labels_group)

        return inputs, targets


    def _group_image_ids(self):
        """ Group img_ids according to batch_size and group_method """
        img_ids = self.data.list_image_index()

        if self.shuffle:
            random.shuffle(img_ids)

        num_groups = math.ceil(len(img_ids) / self.batch_size)

        groups = []
        for group_i in range(num_groups):
            start = self.batch_size * group_i
            end   = self.batch_size * (group_i + 1)
            groups.append(img_ids[start:end])

        return groups

    def _make_index_generator(self):
        """ Returns a generator which yields group index to train the model in """
        # start group_index at -1 so that first group_index returned is 0
        num_groups = len(self.groups)
        group_index = -1
        reset_point = num_groups - 1

        while True:
            if group_index < reset_point:
                group_index += 1
            else:
                if self.shuffle:
                    self.groups = self._group_image_ids()
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

        Generator would return original images and labels along with network inputs.

        Generator generates items in the following format
        [network_inputs, orig_image], target_labels
        """

        img_ids = self.data.list_image_index()
        num_img = len(img_ids)
        i = 0

        while i < num_img:
            img_id = img_ids[i]

            # load image, bbox and labels
            image_group  = self.load_image_group([img_id])
            bbox_group   = self.load_bbox_group([img_id])
            labels_group = self.load_labels_group([img_id])

            image  = image_group[0]
            bbox   = bbox_group[0]
            labels = labels_group[0]

            # Crop image
            if bbox.shape == (4,):
                # Transform bbox into the form [x1, y1, x2, y2, class]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                # Get cropped image
                image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Make copy of original cropped image and labels
            orig_image = image.copy()
            orig_labels = labels

            # resize image
            image = self.resize_image(image)

            # compuate network inputs
            inputs = self.compute_inputs([image])

            # Increase counter
            i += 1

            yield [inputs, orig_image], orig_labels
