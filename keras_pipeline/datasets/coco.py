import os
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from ._ImageDatasetTemplate import ImageDatasetTemplate
from ..preprocessing.image import read_image

class COCODataset(ImageDatasetTemplate):
    """ COCO dataset API meant to be used by generators.DetectionGenerator
    Requires your coco dataset to be setup in the following format

    root_dir
      |- images
      |- annotations

    Args
        root_dir : see above
        set_name : name of the set eg. train2017

    """

    def __init__(self, root_dir, set_name):
        self.root_dir = root_dir
        self.set_name = set_name

        # Load annotation file using pycocotools.coco.COCO
        # Too bad there is no quite mode for this
        if self.set_name == 'test2017':
            annotation_file = os.path.join(
                self.root_dir,
                'annotations',
                'image_info_test2017.json'
            )
        else:
            annotation_file = os.path.join(
                self.root_dir,
                'annotations',
                'instances_{}.json'.format(self.set_name)
            )
        coco = COCO(annotation_file)

        # Retrieve object class information
        # Here object_classes_id  is like id to class_name
        #      object_classes     is like class_name to id
        # It is also important that id number starts from 0 and ends at num_object_classes
        coco_id_to_id          = {}
        self.object_classes_id = {}
        self.object_classes    = {}
        for id, class_info in enumerate(coco.cats.values()):
            coco_id_to_id[class_info['id']] = id
            class_info['id'] = id
            self.object_classes_id [id]             = class_info
            self.object_classes[class_info['name']] = class_info

        # Store image information which also includes the associated annotations
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

            # Edit and store annotations
            image_annotations = coco.imgToAnns[image_index]
            for i in range(len(image_annotations)):
                coco_id = image_annotations[i]['category_id']
                image_annotations[i]['category_id'] = coco_id_to_id[coco_id]
            image_info['annotations'] = image_annotations

            # Calculate aspect_ratio
            image_info['aspect_ratio'] = image_info['width'] / image_info['height']

            # Update in self.image_infos
            self.image_infos[image_index] = image_info


    def list_image_index(self):
        return list(self.image_infos.keys())

    def get_size(self):
        return len(self.image_infos)

    def get_image_aspect_ratio(self, image_index):
        return self.image_infos[image_index]['aspect_ratio']

    def get_num_object_classes(self):
        return len(self.object_classes)

    def load_image(self, image_index):
        file_path = self.image_infos[image_index]['file_path']
        return read_image(file_path)

    def load_image_info(self, image_index):
        return self.image_infos[image_index]

    def load_annotations(self, image_index):
        annotations = self.image_infos[image_index]['annotations']
        # some annotations have no width or height, skip them
        return [ann for ann in annotations if (ann['bbox'][2] >= 1) and (ann['bbox'][3] >= 1)]

    def load_annotations_array(self, image_index):
        annotations = self.load_annotations(image_index)
        if len(annotations) > 0:
            return np.array([ann['bbox'] + [ann['category_id']] for ann in annotations])
        else:
            return np.zeros((0, 5))

    def object_class_to_object_class_id(self, name):
        return self.object_classes[name]['id']

    def object_class_id_to_object_class(self, id):
        return self.object_classes_id[id]['name']
