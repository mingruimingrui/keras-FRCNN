import keras


def ResNet50(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a resnet 50 model """
    resnet_model = keras.applications.resnet50.ResNet50(input_tensor=input_tensor, include_top=include_top)

    for layer in resnet_model.layers:
        if 'bn' in layer.name:
            layer.trainable = False

    if freeze_backbone:
        for layer in resnet_model.layers:
            layer.trainable = False

    return resnet_model


def ResNet50Backbone(input_tensor, freeze_backbone=False):
    """ Loads a resnet 50 model as a backbone """
    resnet_model = ResNet50(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    activation_layer_names  = [l.name for l in resnet_model.layers if 'activation' in l.name]

    C1 = resnet_model.get_layer(activation_layer_names[ 0]).output
    C2 = resnet_model.get_layer(activation_layer_names[ 9]).output
    C3 = resnet_model.get_layer(activation_layer_names[21]).output
    C4 = resnet_model.get_layer(activation_layer_names[39]).output
    C5 = resnet_model.get_layer(activation_layer_names[48]).output

    backbone = keras.Model(
        inputs = resnet_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'resnet_backbone'
    )

    return backbone
