""" Many of the datagetters here are written in a shoddy manner
If you are free and is willing to help out you may update this page,
just keep the functions functioning the same
"""

import os
import json
import numpy as np

from keras_pipeline.datasets._ImageDatasetTemplate import ImageDatasetTemplate
from keras_pipeline.preprocessing.image import read_image


def _get_object_classes_and_labels(tag_label_object):
    all_object_classes = {}
    all_object_labels  = {}
    old_tag_label_to_new_label = {}

    label_counter = 0
    for tag_name in ['blur', 'expression', 'illumination', 'occlusion', 'pose']:
        labels = tag_label_object[tag_name]
        old_label_to_new_label = {}

        for index in range(len(labels)):
            label_info = labels[str(index)]

            old_label_to_new_label[label_info['id']] = label_counter

            label_info['id']    = label_counter
            label_info['label'] = label_info['name']
            label_info['name']  = tag_name + '_' + label_info['label']

            all_object_classes[label_info['name']] = label_info
            all_object_labels[label_info['id']]    = label_info

            label_counter += 1

        old_tag_label_to_new_label[tag_name] = old_label_to_new_label

    return all_object_classes, all_object_labels, old_tag_label_to_new_label

def _reformat_image_info(image_infos, old_tag_label_to_new_label):
    """ This currently filters annotations and reformats image_id """
    new_image_infos = {}

    for current_id in image_infos.keys():
        # Get image info for current_id
        current_info = image_infos[current_id]

        # First filter annotations
        current_info['annotations'] = [ann for ann in current_info['annotations'] if ann['invalid'] == 0]
        for i in range(len(current_info['annotations'])):
            category_id = []
            ann = current_info['annotations'][i]

            for tag_name in ['blur', 'expression', 'illumination', 'occlusion', 'pose']:
                category_id.append(old_tag_label_to_new_label[tag_name][ann[tag_name]])

            current_info['annotations'][i]['category_id'] = category_id

        # Now to reformat image_id
        # Get even and file path
        event     = current_info['event']
        file_path = current_info['file_path']

        # Extract data from them to forge new_id
        event_id = int(event.split('-')[0])
        img_number = int(os.path.splitext(file_path)[0].split('_')[-1])

        new_id = 10000 * event_id + img_number

        # store infor with new id in new_iamge_infos
        current_info['id'] = new_id
        new_image_infos[new_id] = current_info

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
        self.object_classes, self.object_labels, old_tag_label_to_new_label = \
            _get_object_classes_and_labels(wider['labels'])

        # Now get image infos
        self.image_infos = _reformat_image_info(wider['image_infos'], old_tag_label_to_new_label)

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
        return len(self.object_classes)

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
        raise NotImplementedError('load_annotations_array method not implemented')

    def object_class_to_label(self, name):
        # label is like an index needs to range from 0 to num_class
        raise NotImplementedError('object_class_to_label method not implemented')

    def label_to_object_class(self, label):
        # label is like an index needs to range from 0 to num_class
        raise NotImplementedError('label_to_object_class method not implemented')
