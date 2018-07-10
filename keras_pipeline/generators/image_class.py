""" Implemented with reference from NumpyArrayIterator at
https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""

import math
import numpy as np
import threading

import keras

from ._image_generator import ImageGenerator

from ..preprocessing.image_transform import (
    adjust_transform_for_image,
    apply_transform,
    resize_image_2
)


class ImageClassGenerator(ImageGenerator):
    def __init__(self, config):
        # Store general dataset info
        self.dataset         = config.dataset
        self.all_image_index = config.dataset.list_all_image_index()
        self.size            = config.dataset.get_size()
        self.num_classes     = config.dataset.get_num_classes()
        self.label_to_name   = config.dataset.label_to_name

        # Typical generator config
        self.batch_size      = config.batch_size
        self.image_height    = config.image_height
        self.image_width     = config.image_width
        self.stretch_to_fill = config.stretch_to_fill
        self.shuffle         = config.shuffle

        # Create transform generator
        self.transform_parameters = config.transform_parameters
        self.transform_generator = None
        if config.allow_transform:
            self.transform_generator = self._make_transform_generator(config)

        # Validate dataset
        self._validate_dataset()

        # Tools which helps order the data generated
        self.lock = threading.Lock() # this is to allow for parrallel batch processing
        self.group_index_generator = self._make_index_generator()

    def _validate_dataset(self):
        """ Dataset validator which validates the suitability of the dataset """
        img_id = self.dataset.list_all_image_index()[0]

        size         = self.dataset.get_size()
        num_classes  = self.dataset.get_num_classes()
        assert isinstance(size, int), 'Size must be an integer, got {}'.format(type(size))
        assert isinstance(num_classes, int), 'Num classes must be an integer, got {}'.format(type(num_classes))

        img = self.dataset.load_image(img_id)
        assert (len(img.shape) == 3) and (img.shape[-1] == 3), 'img is of wrong shape, got {}'.format(img.shape)

        bbox = self.dataset.get_image_bbox_array(img_id)
        assert bbox.shape == (4,), 'bbox generated must be of shape (4,), got {}'.format(bbox.shape)

        labels = self.dataset.get_image_class_array(img_id)
        assert len(labels.shape) == 1, 'get_image_class_array should return a 1-D array, got {}'.format(labels.shape)

        self.dataset.label_to_name(num_classes - 1)

    def load_X_group(self, group):
        """ Loads the raw group images and bboxes from the dataset """
        return [(self.dataset.load_image(image_index), self.dataset.get_image_bbox_array(image_index))
            for image_index in group]

    def load_Y_group(self, group):
        """ Loads the raw inputs from the dataset """
        return [self.dataset.get_image_class_array(image_index) for image_index in group]

    def random_transform_image(self, image):
        """ Apply a random transformation on an image """
        transformation = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
        image = apply_transform(transformation, image, self.transform_parameters)
        return image

    def resize_image(self, image):
        return resize_image_2(
            image,
            width=self.image_width,
            height=self.image_height,
            stretch_to_fill=self.stretch_to_fill
        )

    def preprocess_entry(self, X, labels):
        """ Preprocesses an entry """
        (image, bbox) = X

        # Crop image
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Apply transformation
        if self.transform_generator:
            image = self.random_transform_image(image)

        # resize image
        image = self.resize_image(image)

        return image, labels

    def compute_inputs(self, image_group):
        """ Compute the network inputs """
        # Construct an image batch object
        batch_shape = (self.batch_size, self.image_height, self.image_width, 3)
        image_batch = np.zeros(batch_shape, dtype=keras.backend.floatx())

        # Copy all images to the center of the image batch object
        for image_index, image in enumerate(image_group):
            x1 = math.floor((self.image_width  - image.shape[1]) / 2)
            y1 = math.floor((self.image_height - image.shape[0]) / 2)
            x2 = x1 + image.shape[1]
            y2 = y1 + image.shape[0]
            image_batch[image_index, y1:y2, x1:x2] = image

        return image_batch


    def compute_targets(self, labels_group):
        """ Compute the network outputs """
        # Construct a labels batch object
        batch_shape  = (self.batch_size, self.num_classes)
        labels_batch = np.zeros(batch_shape, dtype=keras.backend.floatx())

        # Copy all label info to the batch object
        for labels_index, labels in enumerate(labels_group):
            labels_batch[labels_index, labels] = 1

        return labels_batch
