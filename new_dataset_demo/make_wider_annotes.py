import os
import sys
import json
from tqdm import tqdm

import numpy as np
from scipy.io import loadmat

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

from keras_pipeline.preprocessing.image import read_image


ROOTDIR  = 'D://datasets/wider'
ANNDIR   = os.path.join(ROOTDIR, 'annotations')
IMGDIR   = os.path.join(ROOTDIR, 'images')
DATASETS = ['val', 'train', 'test']


def get_labels():
    return {
        'blur': {
            0: {'id': 0, 'name': 'clear' , 'tag': 'blur'},
            1: {'id': 1, 'name': 'normal', 'tag': 'blur'},
            2: {'id': 2, 'name': 'heavy' , 'tag': 'blur'}
        },
        'expression': {
            0: {'id': 0, 'name': 'typical'   , 'tag': 'expression'},
            1: {'id': 1, 'name': 'exaggerate', 'tag': 'expression'}
        },
        'illumination': {
            0: {'id': 0, 'name': 'normal' , 'tag': 'expression'},
            1: {'id': 1, 'name': 'extreme', 'tag': 'expression'}
        },
        'occlusion': {
            0: {'id': 0, 'name': 'none' , 'tag': 'illumination'},
            1: {'id': 1, 'name': 'partial', 'tag': 'illumination'},
            2: {'id': 2, 'name': 'heavy' , 'tag': 'illumination'}
        },
        'pose': {
            0: {'id': 0, 'name': 'typical' , 'tag': 'pose'},
            1: {'id': 1, 'name': 'atypical', 'tag': 'pose'}
        },
        'invalid': {
            0: {'id': 0, 'name': 'false', 'tag': 'invalid'},
            1: {'id': 1, 'name': 'true' , 'tag': 'invalid'}
        },
    }


def get_tags():
    return list(get_labels().keys())


def get_events(data):
    event_list = [str(event[0]) for event in data['event_list'][:, 0]]
    return event_list


def get_num_images(data):
    image_count = 0

    for event_id, event in enumerate(get_events(data)):
        file_list = [str(f[0][0]) for f in data['file_list'][event_id, 0]]
        image_count += len(file_list)

    return image_count


def get_image_infos(data, dataset_name):
    all_image_info = {}
    image_count = 0

    with tqdm(total=get_num_images(data), desc='Processing annotes for {}'.format(dataset_name)) as pbar:

        for event_id, event in enumerate(get_events(data)):
            file_list = [str(f[0][0]) for f in data['file_list'][event_id, 0]]

            for file_id, file_name in enumerate(file_list):
                # Get image info
                image_info = {
                    'id'          : image_count,
                    'event'       : str(event),
                    'file_path'   : str(os.path.join(ROOTDIR, 'images', event, file_name) + '.jpg'),
                    'annotations' : []
                }

                # Read image to get height and width
                image = read_image(image_info['file_path'])

                image_info['height'] = int(image.shape[1])
                image_info['width']  = int(image.shape[0])
                image_info['aspect_ratio'] = image_info['width'] / image_info['height']

                # Store info
                all_image_info[image_count] = image_info

                # Update progress
                image_count += 1
                pbar.update(1)

    return all_image_info


def get_anns(data):
    all_image_to_anns = {}
    image_count = 0
    ann_count = 0

    for event_id, event in enumerate(get_events(data)):
            file_list = [str(f[0][0]) for f in data['file_list'][event_id, 0]]

            for file_id, file_name in enumerate(file_list):
                anns_for_image = []

                for ann_idx, ann in enumerate(data['face_bbx_list'][event_id, 0][file_id, 0]):
                    # Get ann info
                    ann_info = {
                        'id'       : ann_count,
                        'image_id' : image_count,
                        'area'     : int(ann[2] * ann[3]),
                        'bbox'     : [int(x) for x in ann.tolist()]
                    }

                    for tag in get_tags():
                        ann_info[tag] = data[tag + '_label_list'][event_id, 0][file_id, 0][ann_idx, 0]
                        ann_info[tag] = int(ann_info[tag])

                    # Store info
                    anns_for_image.append(ann_info)

                    # Update progress
                    ann_count += 1

                # Store info
                all_image_to_anns[image_count] = anns_for_image

                # Update progress
                image_count += 1

    return all_image_to_anns


def append_anns_to_image(image_infos, image_to_anns):
    for image_id in image_infos.keys():
        image_infos[image_id]['annotations'] = image_to_anns[image_id]

    return image_infos


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    for dataset_name in DATASETS:
        json_filepath = os.path.join(ANNDIR, 'wider_face_{}.json'.format(dataset_name))

        # Load data
        data = loadmat(os.path.join(ANNDIR, 'wider_face_{}.mat'.format(dataset_name)))

        # Create template to store data later
        data_json = {}

        data_json['labels'] = get_labels()
        data_json['tags']   = get_tags()
        data_json['events'] = get_events(data)

        data_json['num_images']  = get_num_images(data)
        data_json['image_infos'] = get_image_infos(data, dataset_name)

        if dataset_name != 'test':
            data_json['image_to_anns'] = get_anns(data)
            data_json['image_infos'] = append_anns_to_image(data_json['image_infos'], data_json['image_to_anns'])

        save_json(data_json, json_filepath)

if __name__ == '__main__':
    main()
