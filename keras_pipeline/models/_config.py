import sys
import keras
from ._config_template import ModelConfigTemplate


class RetinaNetConfig(ModelConfigTemplate):
    """ For help on RetinaNetConfig, use RetinaNetConfig.help() """

    def __init__(self, **kwargs):
        self.__name__ = 'RetinaNetConfig'
        self.__params__ = {}

        self.add(
            'name',
            'Name of your model',
            default = 'retinanet',
            accpted_types = str
        )

        self.add(
            'num_classes',
            'Number of classes',
            accpted_types = 'int-like',
            condition = lambda x: x > 0,
            required = True
        )

        self.add(
            'input_shape',
            'Shape of your input',
            accepted_types = 'list-like',
            condition = lambda x: len(x) == 3
        )

        def is_input_tensor_valid(input_tensor):
            valid_type = type(input_tensor) == type(keras.Input(shape=(1, 1)))
            valid_len  = len(input_tensor) == 3

            return valid_type & valid_len

        self.add(
            'input_tensor',
            'Tensor as an input',
            condition = is_input_tensor_valid
        )

        self.add(
            'backbone_name',
            'Loads a backbone based on a name',
            default = 'inception_v3',
            valid_options = ['inception_v3']
        )

        self.add(
            'freeze_backbone',
            'Freezes the backbone weights during training',
            default = False,
            accepted_types = bool
        )

        self._check_valid(**kwargs)


    def get_input_tensor(self):
        if self.input_tensor is None:
            if self.input_shape is None:
                return keras.Input(shape=(None, None, 3))
            else:
                return keras.Input(shape=self.input_shape)
        else:
            return self.input_tensor


    def get_num_anchors(self):
        return 9
