import numpy as np
import keras
from .. import layers


custom_objects = {
    'ResNetPreprocess' : layers.ResNetPreprocess
}


def VGG16(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a vgg 16 model with preprocessing """
    x = layers.ResNetPreprocess()(input_tensor)
    vgg_model = keras.applications.vgg16.VGG16(input_tensor=x, include_top=include_top)

    if freeze_backbone:
        for layer in vgg_model.layers:
            layer.trainable = False

    return keras.Model(
        inputs = input_tensor,
        outputs = vgg_model.output,
        name = 'vgg16'
    )


def VGG16Backbone(input_tensor, freeze_backbone=False):
    """ Loads a vgg 16 model as a backbone """
    vgg_model = VGG16(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    layer_names = [l.name for l in vgg_model.layers]

    C1 = vgg_model.get_layer(layer_names[ 5]).output
    C2 = vgg_model.get_layer(layer_names[ 9]).output
    C3 = vgg_model.get_layer(layer_names[13]).output
    C4 = vgg_model.get_layer(layer_names[17]).output
    C5 = vgg_model.get_layer(layer_names[18]).output

    return keras.Model(
        inputs  = vgg_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name    = 'vgg_backbone'
    )


def compute_pyramid_feature_shapes_for_img_shape(image_shape):
    C0_shape = np.array(image_shape[:2])

    C1_shape = np.floor(C0_shape / 2)
    C2_shape = np.floor(C1_shape / 2)

    P3_shape = np.floor(C2_shape / 2)
    P4_shape = np.floor(P3_shape / 2)
    P5_shape = np.floor(P4_shape / 2)

    P6_shape = np.floor(P5_shape / 2)
    P7_shape = np.floor(P6_shape / 2)

    return P3_shape, P4_shape, P5_shape, P6_shape, P7_shape
