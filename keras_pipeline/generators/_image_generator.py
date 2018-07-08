""" Implemented with reference from NumpyArrayIterator at
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""

import math
import random
import threading

from ..preprocessing.transform import random_transform_generator


class ImageGenerator(object):
    ###########################################################################
    #### This marks the start of uniquely defined functions

    def __init__(self, config):
        # Store config settings
        # Ensure that atleast the following variable are present
        self.dataset
        self.all_image_index
        self.size
        self.shuffle
        self.batch_size

        raise NotImplementedError('__init__ is not defined')

        # Validate dataset
        self._validate_dataset()

        # Tools which helps order the data generated
        self.lock = threading.Lock() # this is to allow for parrallel batch processing
        self.group_index_generator = self._make_index_generator()

    def _validate_dataset(self):
        """ Dataset validator which validates the suitability of the dataset """

    def load_X_group(self, group):
        """ Loads the raw inputs from the dataset """
        raise NotImplementedError('load_X_group is not defined')

    def load_Y_group(self, group):
        """ Loads the raw inputs from the dataset """
        raise NotImplementedError('load_Y_group is not defined')

    def preprocess_entry(self, X, Y):
        """ Preprocesses an entry """
        raise NotImplementedError('preprocess_entry is not defined')

    def compute_inputs(self, X_group):
        """ Compute the network inputs """
        raise NotImplementedError('compute_inputs is not defined')

    def compute_targets(self, Y_group):
        """ Compute the network outputs """
        raise NotImplementedError('compute_targets is not defined')


    ###########################################################################
    #### This marks the start of helper functions

    def _make_transform_generator(self, config):
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

    def _group_image_ids(self):
        """ Group img_ids according to batch_size """
        # Retrieve all image ids and count number of groups
        img_ids = self.all_image_index
        num_groups = math.ceil(len(img_ids) / self.batch_size)

        # Perform shuffling
        if self.shuffle:
            random.shuffle(img_ids)

        # Group image ids
        groups = []
        for group_i in range(num_groups):
            start = self.batch_size * group_i
            end   = self.batch_size * (group_i + 1)
            groups.append(img_ids[start:end])

        # Save groups
        self.groups = groups


    ###########################################################################
    #### This marks the start of essential functions

    def preprocess_group(self, X_group, Y_group):
        for index, (X, Y) in enumerate(zip(X_group, Y_group)):
            # Preprocess single group entry
            X, Y = self.preprocess_entry(X, Y)

            # Update group
            X_group[index] = X
            Y_group[index] = Y

        return X_group, Y_group

    def _get_batches_of_transformed_samples(self, group):
        # load group X and Y
        X_group = self.load_X_group(group)
        Y_group = self.load_Y_group(group)

        # perform preprocessing
        X_group, Y_group = self.preprocess_group(X_group, Y_group)

        # compuate network inputs
        inputs = self.compute_inputs(X_group)

        # compute network targets
        targets = self.compute_targets(X_group, Y_group)

        return inputs, targets

    def _make_index_generator(self):
        """ Returns a generator which yields group index to train the model in """
        # Initialize a grouping order
        self._group_image_ids()

        # start group_index at -1 so that first group_index returned is 0
        num_groups = len(self.groups)
        group_index = -1
        reset_point = num_groups - 1

        while True:
            if group_index < reset_point:
                group_index += 1
            else:
                if self.shuffle:
                    self._group_image_ids()
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
