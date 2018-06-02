import keras


def InceptionV3(input_tensor, include_top=True, freeze_backbone=False):
    """ Loads an inception v3 model """
    inception_model = keras.applications.inception_v3.InceptionV3(input_tensor=input_tensor, include_top=include_top)

    for layer in inception_model.layers:
        if 'normalization' in layer.name:
            layer.trainable = False

    if freeze_backbone:
        for layer in inception_model.layers:
            layer.trainable = False

    return inception_model


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

    backbone = keras.Model(
        inputs = inception_model.input,
        outputs = [C1, C2, C3, C4, C5],
        name = 'inception_backbone'
    )

    return backbone
