import cv2
import numpy as np

import keras

from ..preprocessing.image_transform import resize_image_1
from .anchors import compute_overlap
from .visualization import draw_detections

def detection_inference_on_single_image(model, image, min_side=800, max_side=1333):
    """ Performs inference on a single image using a detection model

    Args
        model    : Detection model with predict method
        image    : Image in RGB format
        min_side : Minimum side length of model input
        max_side : Maximum side length of model input
    Returns
        boxes    : The bounding box axis for each detection in the format [x1, y1, x2, y2]
        scores   : The scores for each detection
        labels   : The labels for each detection

    """
    assert len(image.shape) == 3, 'Image must be of dimension 3'

    # Make copy of image
    image = image.copy()

    # Resize Image
    image, scale = resize_image_1(image, min_side=min_side, max_side=max_side)

    # Perform prediction on image
    boxes, scores, labels = model.predict(np.expand_dims(image, 0))

    # Correct boxes for scale
    boxes /= scale

    return boxes[0], scores[0], labels[0]


def detection_inference_on_batch(model, image_group, min_side=800, max_side=1333):
    """ Performs inference on a batch of images using a detection model

    Args
        model       : Detection model with predict method
        image_group : List (not array) of images in RGB format
        min_side    : Minimum side length of model input
        max_side    : Maximum side length of model input
    Returns
        boxes  : The bounding box axis for each detection in the format [x1, y1, x2, y2]
        scores : The scores for each detection
        labels : The labels for each detection
    """
    batch_size = len(image_group)
    scales = np.zeros((batch_size, 1))

    for image_index, image in enumerate(image_group):
        assert len(image == 3), 'Each image must be of dimension 3'

        # Make copy of image
        image = image.copy()

        # Resize image
        image, scale = resize_image_1(image, min_side=min_side, max_side=max_side)

        # Store image back into list
        image_group[image_index] = image
        scales[image_index] = scale

    # Find max shape
    max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

    # Create placeholder
    image_batch = np.zeros((batch_size,) + max_shape, dtype=keras.backend.floatx())

    # Store all images into placeholder
    for image_index, image in enumerate(image_group):
        image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

    # Perform prediction on batch
    boxes, scores, labels = model.predict_on_batch(image_batch)

    # Correct boxes against scales
    boxes /= scales

    return boxes, scores, labels
