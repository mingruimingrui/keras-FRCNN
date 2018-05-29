""" Implemented with reference from NumpyArrayIterator at
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""

import math
import numpy as np
import random
import threading

import keras


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

    name = dataset.label_to_object_class(0)
    dataset.object_class_to_label(name)

def _validate_config(config):
    """ Validates that config is suitable for a detection task """
    assert config.compute_anchors is not None, 'For detection and segmentation tasks, compute_anchors is required.'
    assert config.compute_anchors((800, 800, 3))[0].shape == (4,), 'compute_anchors not outputting the correct shape'



class DetectionGenerator(object):
    def __init__(self, config):
        _validate_config(config)
        _validate_dataset(config.dataset)

        self.data            = config.dataset
        self.compute_anchors = config.compute_anchors
        self.batch_size      = config.batch_size
        self.image_min_side  = config.image_min_side
        self.image_max_side  = config.image_max_side
        self.shuffle         = config.shuffle_groups

        # Define groups (1 group ==  1 batch)
        self.groups = _group_image_ids(
            config.dataset.list_image_index(),
            config.batch_size,
            config.group_method,
            config.dataset
        )

        # Create transform generator
        self.transform_generator = None
        if config.allow_transform:
            self.transform_generator = _make_transform_generator(config)

        # Create the tools which helps define the order in which the
        self.lock = threading.Lock() # this is to allow for parrallel batch processing
        self.group_index_generator = self._make_index_generator(len(self.groups))


    def _get_batches_of_transformed_samples(self, group_index):
        # TODO: Generate batch based off group_index
        return group_index, self.groups[group_index], 'TBI'

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
        return self._get_batches_of_transformed_samples(group_index)

    def __len__(self):
        return self.data.get_size()

    def __next__(self):
        return self.next()

    def __iter__(self):
        raise NotImplementedError('Generator is not yet iteratable, still in experimental phase')

        group_index = -1
        reset_point = len(self.groups) - 1

        while group_index < reset_point:
            group_index += 1
            yield self._get_batches_of_transformed_samples(group_index)
