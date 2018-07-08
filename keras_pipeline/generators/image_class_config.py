from collections import OrderedDict

from ..utils._config_template import ConfigTemplate
from ..preprocessing.image_transform import TransformParameters


class ImageClassGeneratorConfig(ConfigTemplate):
    """ For help on ImageClassGeneratorConfig, use ImageClassGeneratorConfig(help=True) """

    def __init__(self, help=False, **kwargs):
        self.__name__ = 'ImageClassGeneratorConfig'
        self.__params__ = OrderedDict()

        # Most frequently used parameters

        self.add(
            'dataset',
            'Dataset for this generator',
            required = True
        )

        self.add(
            'batch_size',
            'Number of images to be processed in parallel during training',
            default = 64,
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
            'image_height',
            'Height of image',
            default = 224,
            accepted_types = 'int-like'
        )

        self.add(
            'image_width',
            'Width of image',
            default = 224,
            accepted_types = 'int-like'
        )

        self.add(
            'stretch_to_fill',
            'Flag to strech image to fill input dimensions',
            default = False,
            accepted_types = bool
        )

        self.add(
            'shuffle',
            'Shuffle the image order',
            default = False,
            accepted_types = bool
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
            default = 0,
            accepted_types = 'numeric'
        )

        if help:
            self.help()
        else:
            self._validate_kwargs_(**kwargs)


    def _validate_kwargs_(self, **kwargs):
        super(ImageClassGeneratorConfig, self)._validate_kwargs_(**kwargs)

        # This is default parameters for misc transformation parameters
        # Eg. padding_mode, image_interpolation, fill_value...
        # There are no plans on adding this to the list of variables available
        self.transform_parameters = TransformParameters()
