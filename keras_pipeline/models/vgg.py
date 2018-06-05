import keras


def VGG16(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads a vgg 16 model """
    vgg_model = keras.applications.vgg16.VGG16(input_tensor=input_tensor, include_top=include_top)

    if freeze_backbone:
        for layer in vgg_model.layers:
            layer.trainable = False

    return vgg_model


def VGG16Backbone(input_tensor, freeze_backbone=False):
    """ Loads a vgg 16 model as a backbone """
    vgg_model = VGG16(input_tensor, include_top=False, freeze_backbone=freeze_backbone)

    layer_names = [l.name for l in vgg_model.layers]

    C1 = vgg_model.get_layer(layer_names[ 5]).output
    C2 = vgg_model.get_layer(layer_names[ 9]).output
    C3 = vgg_model.get_layer(layer_names[13]).output
    C4 = vgg_model.get_layer(layer_names[17]).output
    C5 = vgg_model.get_layer(layer_names[18]).output

    backbone = keras.Model(
        inputs  = vgg_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name    = 'vgg_backbone'
    )

    return backbone
