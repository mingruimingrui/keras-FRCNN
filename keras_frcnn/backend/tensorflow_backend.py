import tensorflow
import keras


def resize_images(*args, **kwargs):
    return tensorflow.image.resize_images(*args, **kwargs)
