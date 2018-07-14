from collections import OrderedDict

from ..utils._config_template import ConfigTemplate
from ..preprocessing.image_transform import TransformParameters


class DetectionGeneratorConfig(ConfigTemplate):
    """ For help on DetectionGeneratorConfig, use DetectionGeneratorConfig(help=True) """

    def __init__(self, help=False, **kwargs):
        self.__name__ = 'DetectionGeneratorConfig'
        self.__params__ = OrderedDict()

        # Most frequently used parameters

        self.add(
            'dataset',
            'Dataset for this generator. Simply have to have the functions: ' + \
            'list_all_image_index, get_size, get_num_classes, get_image_aspect_ratio, ' + \
            'load_image, get_annotations_array, label_to_name',
            required = True
        )

        self.add(
            'model_config',
            'Used to extract model internal structures are needed to generate anchors. ' + \
            'Simply needs to be an object with the attributes anchor_sizes, anchor_strides, ' + \
            'anchor_ratios, anchor_scales and compute_pyramid_feature_shapes_for_img_shape. ' + \
            'Either this variable or all the anchor parameters and ' + \
            'compute_pyramid_feature_shapes_for_img_shape must have values'
        )

        self.add(
            'batch_size',
            'Number of images to be processed in parallel during training recommended 1 for ' + \
            'detection due to the high memory cost',
            default = 1,
            accepted_types = 'int-like'
        )

        self.add(
            'allow_transform',
            'Flag that allows the generator to perform transformation on images',
            default = False,
            accepted_types = bool
        )

        # If not providing model_config
        self.add(
            'anchor_sizes',
            'Required if model_config not provided, used to define dimensions of anchors',
            accepted_types = 'list-like',
            required = True
        )

        self.add(
            'anchor_strides',
            'Required if model_config not provided, used to define dimensions of anchors',
            accepted_types = 'list-like',
            required = True
        )

        self.add(
            'anchor_ratios',
            'Required if model_config not provided, used to define dimensions of anchors',
            accepted_types = 'list-like',
            required = True
        )

        self.add(
            'anchor_scales',
            'Required if model_config not provided, used to define dimensions of anchors',
            accepted_types = 'list-like',
            required = True
        )

        self.add(
            'compute_pyramid_feature_shapes_for_img_shape',
            'Required if model_config not provided, used to define dimensions of anchors',
            required = True
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
            default = False,
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
            default = 0.1,
            accepted_types = 'numeric'
        )

        self.add(
            'min_translation',
            'The minimum translation as 2D column vector',
            default = (-0.1, -0.1),
            accepted_types = 'list-like'
        )

        self.add(
            'max_translation',
            'The maximum translation as 2D column vector',
            default = (0.1, 0.1),
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
        # Remove and process model_config separately
        del self.__params__['model_config']
        model_config = kwargs['model_config']

        # We simply replace default with model_config
        self.__params__['anchor_sizes']  ['default'] = model_config.anchor_sizes
        self.__params__['anchor_strides']['default'] = model_config.anchor_strides
        self.__params__['anchor_ratios'] ['default'] = model_config.anchor_ratios
        self.__params__['anchor_scales'] ['default'] = model_config.anchor_scales
        self.__params__['compute_pyramid_feature_shapes_for_img_shape']['default'] = \
            model_config.compute_pyramid_feature_shapes_for_img_shape

        # Run the standard
        super(DetectionGeneratorConfig, self)._validate_kwargs_(**kwargs)


        # This is default parameters for misc transformation parameters
        # Eg. padding_mode, image_interpolation, fill_value...
        # There are no plans on adding this to the list of variables available
        self.transform_parameters = TransformParameters()
