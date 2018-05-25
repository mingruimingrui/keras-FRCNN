import sys
import numpy as np

from collections import OrderedDict
from ._config_template import ConfigTemplate

import keras
from .. import losses


class RetinaNetConfig(ConfigTemplate):
    """ For help on RetinaNetConfig, use RetinaNetConfig.help() """

    def __init__(self, help=False, **kwargs):
        self.__name__ = 'RetinaNetConfig'
        self.__params__ = OrderedDict()

        # Params most people will look for first

        self.add(
            'name',
            'Name of your model',
            default = 'retinanet',
            accepted_types = str
        )

        self.add(
            'num_classes',
            'Number of classes',
            required = True,
            accepted_types = 'int-like',
            condition = lambda x: x > 0
        )

        self.add(
            'input_shape',
            'Shape of your input',
            accepted_types = 'list-like',
            condition = lambda x: len(x) == 3
        )

        self.add(
            'input_tensor',
            'Tensor as an input (overwrites input_shape)',
            condition = is_input_tensor_valid
        )

        # Loss and optimizer config

        self.add(
            'classification_loss',
            'A classifier loss function for shape (None, None, 4), default focal_loss',
            default = losses.make_detection_focal_loss(),
            condition = lambda x: callable(x)
        )

        self.add(
            'regression_loss',
            'A regression loss function, default smooth_l1_loss\n'            + ' ' * 21 + \
                'Do note that the expected y_true shape is (None, None, 5)\n' + ' ' * 21 + \
                'and expected y_pred shape is (None, None, 4)\n'              + ' ' * 21 + \
                'Last channel in y_true is to determine if anchor should be ignored',
            default = losses.make_detection_smooth_l1_loss(),
            condition = lambda x: callable(x)
        )

        self.add(
            'optimizer',
            'A custom user defined optimizer, default Adam',
            default = keras.optimizers.adam(lr=1e-5, clipnorm=0.001),
            condition = lambda x: issubclass(x.__class__, keras.optimizers.Optimizer)
        )

        # Backbone config

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

        # Model config

        self.add(
            'pyramid_feature_size',
            'Number of features at the pyramid level',
            default = 256,
            accepted_types = 'int-like',
            condition = lambda x: x > 0
        )

        self.add(
            'classification_feature_size',
            'Number of features used in classification layers',
            default = 256,
            accepted_types = 'int-like',
            condition = lambda x: x > 0
        )

        self.add(
            'regression_feature_size',
            'Number of features used in anchor regression layers',
            default = 256,
            accepted_types = 'int-like',
            condition = lambda x: x > 0
        )

        self.add(
            'anchor_sizes',
            'List of size of anchor',
            default = [32, 64, 128, 256, 512],
            accepted_types = 'list-like'
        )

        self.add(
            'anchor_strides',
            'List of strides of anchor (needs to be the same length)',
            default = [8, 16, 32, 64, 128],
            accepted_types = 'list-like'
        )

        self.add(
            'anchor_ratios',
            'List of ratios that anchor are generated wrt a window',
            default = [0.5, 1., 2.],
            accepted_types = 'list-like'
        )

        self.add(
            'anchor_scales',
            'List of scales that anchor are generated wrt a window',
            default = [2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)],
            accepted_types = 'list-like'
        )


        if help:
            self.help()
        else:
            self._validate_kwargs_(**kwargs)


    def _validate_kwargs_(self, **kwargs):
        super(RetinaNetConfig, self)._validate_kwargs_(**kwargs)

        # Anchor strides and
        assert len(self.anchor_sizes) == len(self.anchor_strides)


    def get_input_tensor(self):
        if self.input_tensor is None:
            if self.input_shape is None:
                return keras.Input(shape=(None, None, 3))
            else:
                return keras.Input(shape=self.input_shape)
        else:
            return self.input_tensor


    def get_num_anchors(self):
        return len(self.anchor_ratios) * len(self.anchor_scales)


    def get_feature_pyramid_shapes(self):
        # if self.feature_pyramid_shapes:
        if self.backbone_name is 'inception_v3':
            return [
                np.array([97, 97]),
                np.array([48, 48]),
                np.array([23, 23]),
                np.array([12, 12]),
                np.array([ 6,  6])
            ]
        else:
            raise Exception('{} is invalid backbone_name'.format(self.backbone_name))


def is_input_tensor_valid(input_tensor):
    valid_type = type(input_tensor) == type(keras.Input(shape=(1, 1)))
    valid_len  = len(input_tensor) == 3

    return valid_type & valid_len
