""" Script used to process the kitti dataset annotations into one that is accepted by pycocotools

Please first arrange your kitti dataset

"""

import os
import sys
import json
import argparse

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

from keras_pipeline.preprocessing.image import get_image_size


def parse_kitti_annotation(text):
    """ Parses the annotations from a text and returns in a dictionary form

    Refer to https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    for notes on annotation formats

    """
    ann_components = text.replace('\n', '').split(' ')
    # TODO: Complete this function

    return 'TBI'


def get_kitti_info():
    return {
        'contributor': 'Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago',
        'date_created': '2012',
        'description': 'KITTI Dataset for 2D object detection',
        'url': 'http://www.cvlibs.net/datasets/kitti/',
        'version': '1.0',
        'year': 2012
    }


def get_kitti_licenses():
    return []


def get_kitti_categories(args):
    return [
        {'id': 0, 'name': 'Car'           , 'supercategory': 'Vehicle'           },
        {'id': 1, 'name': 'Van'           , 'supercategory': 'Vehicle'           },
        {'id': 2, 'name': 'Truck'         , 'supercategory': 'Vehicle'           },
        {'id': 3, 'name': 'Pedestrian'    , 'supercategory': 'Person'            },
        {'id': 4, 'name': 'Person_sitting', 'supercategory': 'Person'            },
        {'id': 5, 'name': 'Cyclist'       , 'supercategory': 'Portable_transport'},
        {'id': 6, 'name': 'Tram'          , 'supercategory': 'Portable_transport'}
    ]


def get_kitti_images(args):
    os.listdir()

    # {'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
    #  'date_captured': '2013-11-14 17:02:52',
    #  'file_name': '000000397133.jpg',
    #  'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
    #  'height': 427,
    #  'id': 397133,
    #  'license': 4,
    #  'width': 640}


def get_kitti_annotations(args):
    # {'area': 702.1057499999998,
    #  'bbox': [473.07, 395.93, 38.65, 28.67],
    #  'category_id': 18,
    #  'id': 1768,
    #  'image_id': 289343,
    #  'iscrowd': 0,
    #  'segmentation' ...}


def validate_kitti_dir(kitti_path):
    assert os.path.isdir(kitti_path), 'kitti_path provided is not a path'

    set_names = os.listdir(kitti_path)
    assert 'training' in set_names, 'training set must be in {}'.format(kitti_path)

    for set_name in set_names:
        set_path  = os.path.join(kitti_path, set_name)
        set_files = os.listdir(set_path)
        assert 'image_2' in set_files, 'the folder image_2 must be present in {}'.format(set_path)
        if set_name == 'training':
            assert 'label_2' in set_files, 'the folder label_2 must be present in {}'.format(set_path)


def check_args(args):
    validate_kitti_dir(args.kitti_path)
    return args


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script to preprocess kitti annotations and directories to replicate COCO required formats')

    # Most frequently used params
    parser.add_argument('--kitti-path',
        help='Path to dataset_directory (ie. tmp/KITTI)',
        type=str, required=True)

    return parser.parse_args(args)


def get_args(args):
    return check_args(parse_args(args))


def main():
    args = get_args(sys.argv[1:])

    print('Check clear')





if __name__ == '__main__':
    main()
