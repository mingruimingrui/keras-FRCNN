import numpy as np
import keras
from .. import layers


custom_objects = {
    'ResNetPreprocess' : layers.ResNetPreprocess
}


def ResNet50(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet50 model with preprocessing """
    x = layers.ResNetPreprocess()(input_tensor)
    resnet_model = keras.applications.resnet50.ResNet50(input_tensor=x, include_top=include_top)

    for layer in resnet_model.layers:
        if 'bn' in layer.name:
            layer.trainable = False

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return keras.Model(
        inputs = input_tensor,
        outputs = resnet_model.output,
        name = 'resnet50'
    )


def ResNet50Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet50 model as a backbone """
    resnet_model = ResNet50(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    activation_layer_names  = [l.name for l in resnet_model.layers if 'activation' in l.name]

    C1 = resnet_model.get_layer(activation_layer_names[ 0]).output
    C2 = resnet_model.get_layer(activation_layer_names[ 9]).output
    C3 = resnet_model.get_layer(activation_layer_names[21]).output
    C4 = resnet_model.get_layer(activation_layer_names[39]).output
    C5 = resnet_model.get_layer(activation_layer_names[48]).output

    return keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet50_backbone'
    )


def compute_pyramid_feature_shapes_for_img_shape(image_shape):
    C0_shape = np.array(image_shape[:2])

    C1_shape = np.ceil(C0_shape / 2)
    C2_shape = np.floor(C1_shape / 2)

    P3_shape = np.ceil(C2_shape / 2)
    P4_shape = np.ceil(P3_shape / 2)
    P5_shape = np.ceil(P4_shape / 2)

    P6_shape = np.ceil(P5_shape / 2)
    P7_shape = np.ceil(P6_shape / 2)

    return P3_shape, P4_shape, P5_shape, P6_shape, P7_shape
