import keras
import keras_resnet.models


def ResNet18(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet 18 model """
    resnet_model = keras_resnet.models.ResNet34(input_tensor, include_top=include_top, freeze_bn=True)

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return resnet_model


def ResNet18Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet 18 model as a backbone """
    resnet_model = ResNet18(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    relu_layer_names = [l.name for l in resnet_model.layers if 'relu' in l.name]

    C1 = resnet_model.get_layer(relu_layer_names[0]).output
    C2, C3, C4, C5 = resnet_model.output

    backbone = keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet18_backbone'
    )

    return backbone


def ResNet34(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet 34 model """
    resnet_model = keras_resnet.models.ResNet34(input_tensor, include_top=include_top, freeze_bn=True)

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return resnet_model


def ResNet34Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet 34 model as a backbone """
    resnet_model = ResNet34(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    relu_layer_names = [l.name for l in resnet_model.layers if 'relu' in l.name]

    C1 = resnet_model.get_layer(relu_layer_names[0]).output
    C2, C3, C4, C5 = resnet_model.output

    backbone = keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet34_backbone'
    )

    return backbone


def ResNet50(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet 50 model """
    resnet_model = keras_resnet.models.ResNet50(input_tensor, include_top=include_top, freeze_bn=True)

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return resnet_model


def ResNet50Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet 50 model as a backbone """
    resnet_model = ResNet50(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    relu_layer_names = [l.name for l in resnet_model.layers if 'relu' in l.name]

    C1 = resnet_model.get_layer(relu_layer_names[0]).output
    C2, C3, C4, C5 = resnet_model.output

    backbone = keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet50_backbone'
    )

    return backbone
