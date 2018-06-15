import keras
import keras_resnet.models
from .. import layers


custom_objects = {
    'ResNetPreprocess' : layers.ResNetPreprocess
}
custom_objects.update(keras_resnet.custom_objects)


def ResNet18(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet 18 model with preprocessing """
    x = layers.ResNetPreprocess()(input_tensor)
    resnet_model = keras_resnet.models.ResNet18(keras.Input(tensor=x), include_top=include_top, freeze_bn=True, name='resnet18_')

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return keras.Model(
        inputs = input_tensor,
        outputs = resnet_model(input_tensor),
        name = 'resnet18'
    )


def ResNet18Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet 18 model as a backbone """
    resnet_model = ResNet18(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    relu_layer_names = [l.name for l in resnet_model.layers if 'relu' in l.name]

    C1 = resnet_model.get_layer(relu_layer_names[0]).output
    C2, C3, C4, C5 = resnet_model.output

    return keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet18_backbone'
    )


def ResNet34(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet 34 model with preprocessing """
    x = layers.ResNetPreprocess()(input_tensor)
    resnet_model = keras_resnet.models.ResNet34(keras.Input(tensor=x), include_top=include_top, freeze_bn=True, name='resnet34_')

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return keras.Model(
        inputs = input_tensor,
        outputs = resnet_model(input_tensor),
        name = 'resnet34'
    )


def ResNet34Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet 34 model as a backbone """
    resnet_model = ResNet34(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    relu_layer_names = [l.name for l in resnet_model.layers if 'relu' in l.name]

    C1 = resnet_model.get_layer(relu_layer_names[0]).output
    C2, C3, C4, C5 = resnet_model.output

    return keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet34_backbone'
    )


def ResNet50(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet 50 model with preprocessing """
    x = layers.ResNetPreprocess()(input_tensor)
    resnet_model = keras_resnet.models.ResNet50(keras.Input(tensor=x), include_top=include_top, freeze_bn=True, name='resnet50_')

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return keras.Model(
        inputs = input_tensor,
        outputs = resnet_model(input_tensor),
        name = 'resnet50'
    )


def ResNet50Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet 50 model as a backbone """
    resnet_model = ResNet50(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    relu_layer_names = [l.name for l in resnet_model.layers if 'relu' in l.name]

    C1 = resnet_model.get_layer(relu_layer_names[0]).output
    C2, C3, C4, C5 = resnet_model.output

    return keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet50_backbone'
    )
