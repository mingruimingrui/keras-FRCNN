""" Many of the datagetters here are written in a shoddy manner
If you are free and is willing to help out you may update this page,
just keep the functions functioning the same
"""

import os
import json
import numpy as np

from keras_pipeline.datasets._ImageDatasetTemplate import ImageDatasetTemplate
from keras_pipeline.preprocessing.image import read_image


def _get_object_tags_and_id(wider_annotation_desc):
    # Create empty dictionary of things to return
    all_object_tags    = {}
    all_object_tags_id = {}
    old_tag_id_to_new_id  = {} # supposed to use this like old_tag_id_to_new_id[tag_name][label_id] -> new_id

    id_counter = 0

    # Start of single loop for occlusion
    # TODO: to include everything
    tag_name = 'occlusion'
    tag_desc = wider_annotation_desc[tag_name]
    old_id_to_new_id = {}

    for label_name, label_info in tag_desc.items():
        # We still have to store the old id to new id link to help transform image_info annotations
        old_id_to_new_id[label_info['id']] = id_counter

        # Edit label info to new format
        label_info['id']    = id_counter
        label_info['name']  = '{}_{}'.format(tag_name, label_name)
        label_info['label'] = label_name

        # Streo info
        all_object_tags[label_info['name']]  = label_info
        all_object_tags_id[label_info['id']] = label_info

        id_counter += 1

    old_tag_id_to_new_id['occlusion'] = old_id_to_new_id
    # Start of single loop for occlusion

    return all_object_tags, all_object_tags_id, old_tag_id_to_new_id


def _reformat_image_info(image_infos, old_tag_id_to_new_id):
    """ This currently filters annotations and reformats image_id """
    new_image_id_counter = 0
    new_image_infos = {}

    for current_id in image_infos.keys():
        # Get image info for current_id
        current_info = image_infos[current_id]

        # First filter annotations
        current_info['annotations'] = [ann for ann in current_info['annotations'] if ann['invalid'] == 0]
        for i in range(len(current_info['annotations'])):
            label_ids = []
            ann = current_info['annotations'][i]

            for tag_name in ['occlusion']: # TODO: to include everything
                label_ids.append(old_tag_id_to_new_id[tag_name][ann[tag_name]])

            current_info['annotations'][i]['label_ids'] = label_ids

        # store info with new id in new_image_infos
        current_info['id'] = new_image_id_counter
        assert new_image_id_counter not in new_image_infos, 'image_id will overlap'
        new_image_infos[new_image_id_counter] = current_info
        new_image_id_counter += 1

    return new_image_infos



class WiderDataset(ImageDatasetTemplate):
    """ Wider dataset API meant to be used by generators.DetectionGenerator
    Requires your wider dataset to be setup in the following format

    root_dir
      |- images
      |   |- 0--Parade
      |   |- ...
      |
      |- annotations

    Args
        root_dir : see above
        set_name : name of the set eg. train

    """

    def __init__(self, root_dir, set_name):
        self.root_dir = root_dir
        self.set_name = set_name

        # Load annotation file
        annotation_file = os.path.join(
            self.root_dir,
            'annotations',
            'wider_face_{}.json'.format(self.set_name)
        )
        with open(annotation_file, 'r') as f:
            wider = json.load(f)

        # Retrieve object class information
        # Here object_labels  is like label to class_name
        #      object_classes is like class_name to label
        # It is also important that label number starts from 0 and ends at num_object_classes
        self.object_tags, self.object_tags_id, old_tag_id_to_new_id = \
            _get_object_tags_and_id(wider['labels'])

        # Now get image infos
        self.image_infos = _reformat_image_info(wider['image_infos'], old_tag_id_to_new_id)

        # raise NotImplementedError('Dataset not implemented')


    """ Required functions no matter which kind of task is at hand """
    def list_image_index(self):
        """ Returns list (not numpy array or tuple) of image_index """
        return list(self.image_infos.keys())

    def get_size(self):
        """ Returns number of images in dataset """
        return len(self.image_infos)

    def get_image_aspect_ratio(self, image_index):
        return self.image_infos[image_index]['aspect_ratio']

    def get_num_object_classes(self):
        return len(self.object_tags)

    def load_image(self, image_index):
        file_path = self.image_infos[image_index]['file_path']
        return read_image(file_path)

    def load_image_info(self, image_index):
        """ If this is a classification task this needs to have the following attributes
        category_id : corresponding to the image class labels
        """
        return self.image_infos[image_index]

    def load_annotations(self, image_index):
        """ Returns a list of annotations for image with the image_index
        each annotation needs to have the following attributes
            category_id : corresponding to the object class labels
            bbox        : in [x, y, w, h] format
        """
        annotations = self.image_infos[image_index]['annotations']
        # some annotations have no width or height, skip them
        return [ann for ann in annotations if (ann['bbox'][2] >= 1) and (ann['bbox'][3] >= 1)]

    def load_annotations_array(self, image_index):
        """ Returns the annotations in an array format of shape (num_annotes, 5)
        for each annote, the order should be [x, y, w, h, object_class]
        """
        annotations = self.load_annotations(image_index)
        if len(annotations) > 0:
            return np.array([ann['bbox'] + [ann['label_ids'][0]] for ann in annotations])
        else:
            return np.zeros((0, 5))

    def object_tag_to_object_tag_id(self, name):
        return self.object_tags[name]['id']

    def object_tag_id_to_object_tag(self, id):
        return self.object_tags_id[id]['name']
