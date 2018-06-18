import numpy as np
import keras
from .. import layers


custom_objects = {
    'InceptionPreprocess' : layers.InceptionPreprocess
}


def InceptionV3(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads an inception v3 model with preprocessing """
    x = layers.InceptionPreprocess()(input_tensor)
    inception_model = keras.applications.inception_v3.InceptionV3(input_tensor=x, include_top=include_top)

    for layer in inception_model.layers:
        if 'normalization' in layer.name:
            layer.trainable = False

    if freeze_backbone:
        for layer in inception_model.layers:
            layer.trainable = False

    return keras.Model(
        inputs = input_tensor,
        outputs = inception_model.output,
        name = 'inception_v3'
    )


def InceptionV3Backbone(input_tensor, freeze_backbone=False):
    """ Loads an inception v3 model as a backbone """
    inception_model = InceptionV3(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    mix_layer_names  = [l.name for l in inception_model.layers if 'mixed'  in l.name]
    conv_layer_names = [l.name for l in inception_model.layers if 'conv2d' in l.name]

    C1 = inception_model.get_layer(conv_layer_names[ 1]).output
    C2 = inception_model.get_layer(conv_layer_names[ 4]).output
    C3 = inception_model.get_layer(mix_layer_names [ 2]).output
    C4 = inception_model.get_layer(mix_layer_names [ 7]).output
    C5 = inception_model.get_layer(mix_layer_names [10]).output

    return keras.Model(
        inputs = inception_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'inception_backbone'
    )


def compute_pyramid_feature_shapes_for_img_shape(image_shape):
    C0_shape = np.array(image_shape[:2])

    C1_shape = np.ceil((C0_shape - 2) / 2) - 2
    C2_shape = np.ceil((C1_shape - 2) / 2) - 2

    P3_shape = np.ceil((C2_shape - 2) / 2)
    P4_shape = np.ceil((P3_shape - 2) / 2)
    P5_shape = np.ceil((P4_shape - 2) / 2)

    P6_shape = np.ceil(P5_shape / 2)
    P7_shape = np.ceil(P6_shape / 2)

    return P3_shape, P4_shape, P5_shape, P6_shape, P7_shape
