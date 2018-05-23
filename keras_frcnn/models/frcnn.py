from collections import namedtuple

import numpy as np

import keras
from .. import layers


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
        scales      : LIst of scales to use per location in a feature map

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
    num_anchors = len(ratios) * len(scales)

    return anchors, num_anchors



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


def load_backbone(input_tensor, backbone_name='inception_v3'):
    assert backbone_name in ['inception_v3'], 'Only inception model support currently'

    if backbone_name == 'inception_v3':
        from .inception import inception_backbone
        backbone = inception_backbone(input_tensor)

    return backbone


def FRCNN(
    # More kwargs in the future
    # num_classes,
    input_tensor  = None,
    name          = 'frcnn',
    backbone_name = 'inception_v3'
):
    """build a Faster-RCNN model"""

    if input_tensor is None:
        input_tensor = keras.layers.Input(shape=(None, None, 3))

    backbone = load_backbone(input_tensor, backbone_name=backbone_name)

    # Generate pyramid features
    C3, C4, C5 = backbone.output
    P3, P4, P5, P6, P7 = __build_pyramid_features(C3, C4, C5)
    features = [P3, P4, P5, P6, P7]

    anchors, num_anchors = __build_anchors(features)
    # classification =
    # regression =

    return keras.Model(
        inputs  = input_tensor,
        outputs = [anchors]
    )
    return features, anchors
