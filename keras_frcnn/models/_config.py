import sys
import numpy as np
import keras

class ModelConfigTemplate:
    def __init__(self):

        self.name         = 'HAL'
        """ name
        Description : Name of your model
        Accepted    : String
        """

        self.num_classes  = None
        """ num_classes
        Description : Number of classes
        Accepted    : Int, Int-like eg. np.integer
        """

        self.input_shape  = None
        """ input_shape
        Description : Shape of your input
        Accepted    : list like of length 3, channel last format
        """

        self.input_tensor = None
        """ input_tensor
        Description : Tensor as an input
        Accepted    : Tensor corresponding to keras backend and of length 3, channel last format
        """

        self.backbone_name = 'inception_v3'
        """ backbone_name
        Description : Loads a backbone based on name
        Accpted     : String
        Options     : ['inception_v3']
        """


    def check_valid(self):
        stopifnot(isinstance(self.name, String), 'name is not a string')

        stopifnot(is_int_like(self.num_classes), 'num_classes is not an int \nMAJOR_ERROR: THIS IS A REQUIRED FIELD')

        if self.input_shape is not None:
            stopifnot(is_list_like(self.input_shape), 'input_shape is not list-like')
            stopifnot(len(self.input_shape) == 3, 'input_shape is not length 3')

        if self.input_tensor is not None:
            stopifnot(type(self.input_tensor) == type(keras.Input(shape=(1, 1))), 'input_tensor is not a valid tensor input')
            stopifnot(len(self.input_tensor.shape) == 3, 'input_tensor is not length 3')

        stopifnot(self.backbone_name in ['inception_v3'], "backbone_name is not one of ['inception_v3']")

    def get_input_tensor(self):
        if self.input_tensor is None:
            if self.input_shape is None:
                input_tensor = keras.layers.Input(shape=(None, None, 3))
            else:
                input_tensor = keras.layers.Input(shape=self.input_shape)
        else:
            input_tensor = self.input_tensor

        return input_tensor

    def get_num_anchors(self):
        return 9


def is_int_like(x):
    return np.issubdtype(type(x), np.integer)


def is_list_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def stopifnot(condition, message):
    if not condition:
        sys.exit('MINGRUI_ERROR: {}'.format(message))
