import keras
from .. import layers


def _create_pyramid_features(C1, C2, C3, feature_size=256):
    """Implemented with reference to FPN paper"""

    # First restrict C1, C2, C3
    C1 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced')(C1)
    C2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    C3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)

    # P3 is obtained by applying 3x3 conv on C3
    P3           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(C3)

    # P2 is obtained by concating C3 and C2 and applying 3x3 conv
    C3_upsampled = layers.ResizeTo(name='C3_upsampled')([C3, C2])
    C2           = keras.layers.Concatenate(name='C2_merged')([C3_upsampled, C2])
    P2           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(C2)

    # P1 is obtained by concating C3, C2 and C1 and applying 3x3 conv
    C2_upsampled = layers.ResizeTo(name='C2_upsampled')([C2, C1])
    C1           = keras.layers.Concatenate(name='C1_merged')([C2_upsampled, C1])
    P1           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1')(C1)

    # P4 is obtained by applying 3x3 conv with stride 2 on C3
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P4')(C3)

    # P5 is obtained by applying 3x3 conv with stride 2 on P4
    P5 = keras.layers.Activation('relu', name='P4_relu')(P4)
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P5')(P5)

    return P1, P2, P3, P4, P5


def load_backbone(backbone_name='inception_v3'):
    assert backbone_name in ['inception_v3'], 'Only inception model support currently'

    if backbone_name == 'inception_v3':
        from .inception import inception_backbone
        backbone = inception_backbone()

    return backbone


def FRCNN(
    # More kwargs in the future
    backbone_name='inception_v3'):
    """build a Faster-RCNN model"""

    backbone = load_backbone(backbone_name)
    C1, C2, C3 = backbone.output
    P1, P2, P3, P4, P5 = _create_pyramid_features(C1, C2, C3)

    return P1, P2, P3, P4, P5
