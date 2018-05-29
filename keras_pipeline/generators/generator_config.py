import cv2
from collections import OrderedDict

from ..utils._config_template import ConfigTemplate
from ..datasets import ImageDatasetTemplate
from ..preprocessing.image import TransformParameters


class GeneratorConfig(ConfigTemplate):
    """ For help on GeneratorConfig, use GeneratorConfig(help=True) """

    def __init__(self, help=False, **kwargs):
        self.__name__ = 'GeneratorConfig'
        self.__params__ = OrderedDict()

        # Most frequently used parameters

        self.add(
            'dataset',
            'Dataset for this generator, must be super type ImageDatasetTemplate',
            required = True,
            condition = lambda x: issubclass(type(x), ImageDatasetTemplate)
        )

        self.add(
            'compute_anchors',
            'REQUIRED if task is detection, function that returns the full list of anchors in [x1, y1, x2, y2] given an image_shape'
        )

        self.add(
            'batch_size',
            'Number of images to be processed in parallel during training recommended 32 for classification 1 for detection',
            default = 1,
            accepted_types = 'int-like'
        )

        self.add(
            'allow_transform',
            'Flag that allows the generator to perform transformation on images',
            default = False,
            accepted_types = bool
        )

        # Training input size Parameters

        self.add(
            'image_min_side',
            'Minimum length of side for image',
            default = 800,
            accepted_types = 'int-like'
        )

        self.add(
            'image_max_side',
            'Maximum length of size for image',
            default = 1333,
            accepted_types = 'int-like'
        )

        self.add(
            'shuffle_groups',
            'Shuffle the group order',
            default = True,
            accepted_types = bool
        )

        self.add(
            'group_method',
            'Order to group images recommended ratio to reduce blacked out areas of images',
            default = 'ratio',
            valid_options = [None, 'random', 'ratio']
        )

        # Transform Parameters

        self.add(
            'min_rotation',
            'the minimum rotation in radians for transform as scalar',
            default = -0.1,
            accepted_types = 'numeric'
        )

        self.add(
            'max_rotation',
            'The maximum rotation in radians for transform as scalar',
            default = -0.1,
            accepted_types = 'numeric'
        )

        self.add(
            'min_translation',
            'The minimum translation as 2D column vector',
            default = (0, 0),
            accepted_types = 'list-like'
        )

        self.add(
            'max_translation',
            'The maximum translation as 2D column vector',
            default = (0, 0),
            accepted_types = 'list-like'
        )

        self.add(
            'min_shear',
            'The minimum shear angle in radians',
            default = -0.1,
            accepted_types = 'numeric'
        )

        self.add(
            'max_shear',
            'The maximum shear angle in radians',
            default = 0.1,
            accepted_types = 'numeric'
        )

        self.add(
            'min_scaling',
            'The minimum scaling for transformation as 2D column vector',
            default = (0.9, 0.9),
            accepted_types = 'list-like'
        )

        self.add(
            'max_scaling',
            'The maximum scaling for transformation as 2D column vector',
            default = (1.1, 1.1),
            accepted_types = 'list-like'
        )

        self.add(
            'flip_x_chance',
            'The chance that x-axis will be flipped',
            default = 0.5,
            accepted_types = 'numeric'
        )

        self.add(
            'flip_y_chance',
            'The chance that y-axis will be flipped',
            default = 0.5,
            accepted_types = 'numeric'
        )

        if help:
            self.help()
        else:
            self._validate_kwargs_(**kwargs)

    def _validate_kwargs_(self, **kwargs):
        super(GeneratorConfig, self)._validate_kwargs_(**kwargs)

        # This is default parameters for misc transformation parameters
        # Eg. padding_mode, image_interpolation, fill_value...
        # There are no plans on adding this to the list of variables available
        self.transform_parameters = TransformParameters()
