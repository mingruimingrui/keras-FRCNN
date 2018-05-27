import numpy as np

import keras
from .. import backend
from ..utils import anchors as util_anchors

class Anchors(keras.layers.Layer):
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2], keras.backend.floatx())
        else:
            self.ratios = np.array(self.ratios, keras.backend.floatx())

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
        else:
            self.scales = np.array(self.scales, keras.backend.floatx())

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors     = keras.backend.variable(util_anchors.generate_anchors(
            base_size = self.size,
            ratios    = self.ratios,
            scales    = self.scales
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        # get height and with as well as number of images
        input_shape = keras.backend.shape(inputs)[:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = backend.shift(input_shape[1:3], self.stride, self.anchors)
        # anchors = backend.shift(inputs.shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (input_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None , 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config


class RegressBoxes(keras.layers.Layer):
    "Applies regression on generated anchors"

    def __init__(self, mean=None, std=None, *args, **kwargs):
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class ClipBoxes(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())

        x1 = backend.clip_by_value(boxes[:, :, 0], 0, shape[2])
        y1 = backend.clip_by_value(boxes[:, :, 1], 0, shape[1])
        x2 = backend.clip_by_value(boxes[:, :, 2], 0, shape[2])
        y2 = backend.clip_by_value(boxes[:, :, 3], 0, shape[1])

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
