import cv2
import numpy as np

from ..preprocessing.image_transform import resize_image_1
from .anchors import compute_overlap
from .visualization import draw_detections

def detection_inference_on_single_image(model, img, min_side=800, max_side=1333):
    assert len(img.shape) == 3, 'Image must be of dimension 3'

    # Make copy of original image
    orig_img = img.copy()

    # Resize Image
    img, scale = resize_image_1(img, min_side=min_side, max_side=max_side)

    # Perform prediction on image
    boxes, scores, labels = model.predict(np.expand_dims(img, 0))

    # Correct boxes for scale
    boxes /= scale

    # To draw detections
    # draw_detections(orig_img, boxes[0], scores[0], labels[0])
    # cv2.imwrite(os.path.join(save_path, 'temp.png'), image)

    return orig_img, boxes[0], scores[0], labels[0]
