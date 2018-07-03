import os
import numpy as np

from pycocotools.coco import COCO
from ._ImageDatasetTemplate import ImageDatasetTemplate
from ..preprocessing.image import read_image

class LandmarkDataset(ImageDatasetTemplate):
    """ Dataset API meant to be used by generators.ImageClassGenerator
    Leverages off pycocotools, you will have to provide an appropriate
    annotation file. Also requires your dataset to be setup in the
    following format.

    Args
        root_dir : see above
        ann_file_name: relative path of annotation file from root_dir

    """

    def __init__(self, root_dir, ann_file_name):
        self.root_dir      = root_dir
        self.ann_file_name = ann_file_name

        # Load annotation file using pycocotools.coco.COCO
        # Too bad there is no quiet mode for this
        annotation_file = os.path.join(
            self.root_dir,
            self.ann_file_name
        )
        coco = COCO(annotation_file)

        # Store image information which contains the image paths as well as crop boxes
        self.image_infos = coco.imgs
        for image_index in coco.getImgIds():
            # Retrieve image specific information
            image_info = self.image_infos[image_index]

            # Make full file_path for easy access later
            image_info['file_path'] = os.path.join(
                self.root_dir,
                image_info['file_name']
            )

            # Calculate aspect_ratio
            image_info['aspect_ratio'] = image_info['width'] / image_info['height']

            # Update in self.image_infos
            self.image_infos[image_index] = image_info


    def list_image_index(self):
        return list(self.image_infos.keys())

    def get_size(self):
        return len(self.image_infos)

    def get_num_landmarks(self):
        return len(self.image_infos['n_points'])

    def get_image_aspect_ratio(self, image_index):
        return self.image_infos[image_index]['aspect_ratio']

    def load_image_info(self, image_index):
        return self.image_infos[image_index]

    def load_image(self, image_index):
        file_path = self.image_infos[image_index]['file_path']
        return read_image(file_path)

    def load_image_bbox_array(self, image_index):
        return np.array(self.image_infos[image_index]['bbox'])

    def load_landmark_array(self, image_index):
        return np.array(self.image_infos[image_index]['landmarks'])
