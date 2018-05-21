import keras

def inception_backbone():
    """Loads an inception v3 model"""

    inception_model = keras.applications.inception_v3.InceptionV3(include_top=False, input_shape=(299,299, 3))

    layer_names = [l.name for l in inception_model.layers]
    mix_layers = list(filter(lambda l: 'mixed'      in l, layer_names))

    C1 = inception_model.get_layer(mix_layers[ 2]).output
    C2 = inception_model.get_layer(mix_layers[ 7]).output
    C3 = inception_model.get_layer(mix_layers[10]).output

    backbone = keras.Model(
        inputs = inception_model.input,
        outputs = [C1, C2, C3]
    )

    return backbone
