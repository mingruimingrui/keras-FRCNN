from collections import namedtuple

import numpy as np

import keras
from .. import layers


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size        = 256,
    classification_feature_size = 256,
    prior_probability           = 0.01,
    name                        = 'classification_submodel'
):
    """Creates a default classification model

    Args
        num_classes                 : Number of classes to predict a score for at each feature level
        num_anchors                 : Number of anchors to predict classification scores for at each feature level
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels
        classification_feature_size : The number of filters to use in the layers in the classification submodel
        name                        : The name of the submodel

    Returns
        A keras.models.Model that predicts classes for each anchor

    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=keras.initializers.Constant(value=-np.log((1 - prior_probability) / prior_probability)),
        # bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(
    num_anchors,
    pyramid_feature_size=256,
    regression_feature_size=256,
    name='regression_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_anchors             : Number of anchors to regress for each feature level
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels
        regression_feature_size : The number of filters to use in the layers in the regression submodel
        name                    : The name of the submodel

    Returns
        A keras.models.Model that predicts regression values for each anchor

    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __apply_model(model, features, name):
    """Applies a single submodel to each FPN level

    Args
        model    : The submodel to evaluate
        features : The FPN features
        name     : Name of the submodel

    Returns
        A tensor containing the response from the submodel on the FPN features

    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_anchors(
    features,
    sizes   = [32, 64, 128, 256, 512],
    strides = [ 8, 16,  32,  64, 128],
    ratios  = [0.5, 1., 2.],
    scales  = [2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)],
):
    """Builds anchors based on shape of features

    Args
        features    : FPN pyramid features
        sizes       : List of sizes to use. Each size corresponds to one feature level
        strides     : List of strides to use. Each stride corresponds to one feature level
        ratios      : List of ratios to use per location in a feature map
        scales      : List of scales to use per location in a feature map

    Returns
        anchors     : Tensor representing anchors generated from features
        num_anchors : Number of anchors at each feature level based on your ratios and scales

    """

    assert len(features) == len(sizes), 'Must have 5 anchor sizes'
    assert len(features) == len(strides), 'Must have 5 anchor strides'

    anchors = []
    for i, f in enumerate(features):
        anchor = layers.Anchors(
            size   = sizes[i],
            stride = strides[i],
            ratios = ratios,
            scales = scales,
            name   = 'anchors_{}'.format(i)
        )(f)
        anchors.append(anchor)

    anchors = keras.layers.Concatenate(axis=1, name='anchors')(anchors)

    return anchors


def __build_pyramid_features(C3, C4, C5, feature_size=256):
    """Builds a pyramid of features
    Implemented with reference to FPN paper

    Args
        C3, ..., C5 : Outputs of different levels from backbone

    Returns
        P3, ..., P7 : Tensor representing features of different levels

    """

    # First restrict C3, C4, C5
    C3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    C4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    C5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)

    # P5 is obtained by applying 3x3 conv on C5
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(C5)

    # P4 is obtained by concating C5 and C4 and applying 3x3 conv
    C5_upsampled = layers.ResizeTo(name='C5_upsampled')([C5, C4])
    C4           = keras.layers.Concatenate(name='C4_merged')([C5_upsampled, C4])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(C4)

    # P3 is obtained by concating C5, C4 and C3 and applying 3x3 conv
    C4_upsampled = layers.ResizeTo(name='C4_upsampled')([C4, C3])
    C3           = keras.layers.Concatenate(name='C3_merged')([C4_upsampled, C3])
    P3           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(C3)

    # P6 is obtained by applying 3x3 conv with stride 2 on C5
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # P7 is obtained by applying 3x3 conv with stride 2 on P6
    P7 = keras.layers.Activation('relu', name='P6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7


def load_backbone(input_tensor, backbone_name='inception_v3', freeze_backbone=False):
    assert backbone_name in ['inception_v3'], 'Only inception model support currently'

    if backbone_name == 'inception_v3':
        from .inception import InceptionBackbone
        backbone = InceptionBackbone(input_tensor, freeze_backbone=freeze_backbone)

    return backbone


def RetinaNetTrain(config):
    """ Build a retinanet model for training

    Args
        config - A RetinaNetConfig object, refer to
                 keras_pipeline.models.RetinaNetConfig(num_classes=1).help()

    Returns
        A retinanet model that returns classification and regression for your defined anchors

    """
    # Get input_tensor
    input = config.input_tensor

    # Generate pyramid features
    backbone = load_backbone(config.input_shape, backbone_name=config.backbone_name, freeze_backbone=config.freeze_backbone)
    _, _, C3, C4, C5 = backbone(input)
    features = __build_pyramid_features(C3, C4, C5, feature_size=config.pyramid_feature_size)

    # Create classification and regression models
    classification_model = default_classification_model(
        num_classes                 = config.num_classes,
        num_anchors                 = config.get_num_anchors(),
        pyramid_feature_size        = config.pyramid_feature_size,
        classification_feature_size = config.classification_feature_size
    )

    regression_model = default_regression_model(
        num_anchors             = config.get_num_anchors(),
        pyramid_feature_size    = config.pyramid_feature_size,
        regression_feature_size = config.regression_feature_size
    )

    # Build anchors and calculate classification and regression
    classification = __apply_model(classification_model, features, name='classification')
    regression     = __apply_model(regression_model    , features, name='regression'    )

    # Build model
    training_model = keras.Model(
        inputs  = input,
        outputs = [classification, regression],
        name    = config.name
    )

    training_model.compile(
        loss = {
            'classification': config.classification_loss,
            'regression'    : config.regression_loss
        },
        optimizer = config.optimizer
    )

    return training_model


def RetinaNetFromTrain(model, config):
    """ Build a retinanet model for inference from a training model

    Args
        model  - A RetinaNetTrain model
        config - A RetinaNetConfig object, refer to
                 keras_pipeline.models.RetinaNetConfig(num_classes=1).help()

    Returns
        A retinanet model that returns detections

    """

    # Extract input and pyramid level features from the model
    input = model.input
    features = [model.get_layer(l).output for l in ('P3', 'P4', 'P5', 'P6', 'P7')]

    # Get classification, regression and anchors
    classification, regression = model.output
    anchors = __build_anchors(
        features,
        sizes   = config.anchor_sizes,
        strides = config.anchor_strides,
        ratios  = config.anchor_ratios,
        scales  = config.anchor_scales,
    )

    # Apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([input, boxes])

    # Calculate detections
    detections = layers.FilterDetections(name='nms')([boxes, classification])

    # Define outputs
    outputs = detections

    return keras.Model(
        inputs  = input,
        outputs = outputs,
        name    = config.name
    )


def RetinaNet(config):
    """ Build a retinanet model for inference

    Args
        config - A RetinaNetConfig object, refer to
                 keras_pipeline.models.RetinaNetConfig(num_classes=1).help()

    Returns
        A retinanet model that returns detections

    """
    # This maybe seems unintuitive, but the prediction model is built on top a of a train model
    training_model = RetinaNetTrain(config)
    prediction_model = RetinaNetFromTrain(training_model, config)

    return prediction_model
