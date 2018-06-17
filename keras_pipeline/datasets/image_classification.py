import os
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from ._ImageDatasetTemplate import ImageDatasetTemplate
from ..preprocessing.image import read_image

class ImageClassificationDataset(ImageDatasetTemplate):
    """ API meant to be used by generators.ImageClassificationGenerator
    Requires your dataset to be setup in the following format and your annotation
    file to follow COCO like convention

    root_dir
      |- images
      |  |- <set_1>
      |  |- <set_2>
      |  |- <set_3>
      |  |- ...
      |- annotations

    Args
        root_dir : see above
        set_name : name of the set eg. train2017

    """

    def __init__(self, root_dir, ann_file_name, set_name):
        self.root_dir      = root_dir
        self.ann_file_name = ann_file_name
        self.set_name      = set_name

        # Load annotation file using pycocotools.coco.COCO
        # Too bad there is no quiet mode for this
        annotation_file = os.path.join(
            self.root_dir,
            'annotations',
            self.ann_file_name
        )
        coco = COCO(annotation_file)

        # Retrieve image class information
        # Here image_classes_id  is like id to class_name
        #      image_classes     is like class_name to id
        # It is also important that id number starts from 0 and ends at num_image_classes
        coco_id_to_id         = {}
        self.image_classes_id = {}
        self.image_classes    = {}
        for id, class_info in enumerate(coco.cats.values()):
            coco_id_to_id[class_info['id']] = id
            class_info['id'] = id
            self.image_classes_id [id]             = class_info
            self.image_classes[class_info['name']] = class_info

        # Append useful image informations such as aspect_ratio
        self.image_infos = coco.imgs
        for image_index in coco.getImgIds():
            # Retrieve image specific information
            image_info = self.image_infos[image_index]

            # Make full file_path for easy access later
            image_info['file_path'] = os.path.join(
                self.root_dir,
                'images',
                self.set_name,
                image_info['file_name']
            )

            # Edit and store category_ids
            import pdb; pdb.set_trace()
            # DEBUG

            # Calculate aspect_ratio
            image_info['aspect_ratio'] = image_info['width'] / image_info['height']

            # Update in self.image_infos
            self.image_infos[image_index] = image_info
