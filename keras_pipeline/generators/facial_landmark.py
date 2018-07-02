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

from ..preprocessing.transform import (
    random_transform_generator,
    transform_xy
)

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
    dataset.get_image_aspect_ratio(img_id)

    dataset.load_image(img_id)
    dataset.load_image_info(img_id)

    dataset.load_image_bbox_array(img_id)
    dataset.load_facial_landmark_array(img_id)


class FacialLandmarkGenerator(object):
    def __init__(self, config):
        _validate_dataset(config.dataset)

        # here dataset is the only object attribute
        # generally object and callable attributes in the generator is
        # ill adviced due to to their sometimes unpickable nature
        # The DetectionDataset object is picklable
        self.data            = config.dataset
        self.size            = config.dataset.get_size()

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
        labels will be in the form [n_points, 2] """
        return [self.data.load_facial_landmark_array(image_index) for image_index in group]

    ###########################################################################
    #### This marks the start of _get_batches_of_transformed_samples helpers

    def random_transform_entry(image, labels):
        # TODO: DO ME TONIGHT!
        return image, labels

    def preprocess_entry(image, bbox, labels):
        if bbox.shape == (4,):
            # Transform bbox into the form [x1, y1, x2, y2, class]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox = [int(b) for b in bbox]

            # Get cropped image
            image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # TBI CONFIRM
            labels[:, 0] -= bbox[0]
            labels[:, 1] -= bbox[1]

        # Apply transformation
        if self.transform_generator:
            image, labels = self.random_transform_entry(image, labels)

        # resize image
        # image = self.resize_image(image)

        return, image, labels

    def preprocess_image_group(self, image_group, bbox_group, labels_group):
        for i, (image, bbox, labels) in enumerate(zip(image_group, bbox_group, labels_group)):
            # Preprocess each image individually
            image, labels = preprocess_entry(image, bbox, labels)

            # Store preprocessed image and labels back into group
            image_group[i]  = image
            labels_group[i] = labels

        return image_group, labels_group

    ###########################################################################
    #### This marks the end of _get_batches_of_transformed_samples helpers

    def _get_batches_of_transformed_samples(self, group):
        # load images bboxes, and labels
        image_group  = self.load_image_group(group)
        bbox_group   = self.load_bbox_group(group)
        labels_group = self.load_labels_group(group)

        import pdb; pdb.set_trace()

        # perform preprocessing on image
        image_group, labels_group = self.preprocess_image_group(image_group, bbox_group, labels_group)

        # # compuate network inputs
        # inputs = self.compute_inputs(image_group)
        #
        # # compute network targets
        # targets = self.compute_targets(labels_group)
        #
        # return inputs, targets

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
        return self.size

    def __next__(self):
        return self.next()
