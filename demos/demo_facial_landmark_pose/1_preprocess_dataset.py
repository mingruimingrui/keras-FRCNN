""" Script used to process the 300W dataset into a form consistent with the coco annotation style

Please first arrange your coco dataset in this order under your coco root directory

"""

import os
import sys
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

from keras_pipeline.preprocessing.image import get_image_size


SETTINGS = ['01_Indoor', '02_Outdoor']


def parse_pts_file(file_path):
    """ Reads a .pts file and outputs a dictionary containing the points information
    Args
        file_path : path to the .pts file
    Returns
        dict containing n_points as well as the points
    """

    assert file_path.endswith('.pts'), 'Must provide a .pts file'

    # Load contents of file
    with open(file_path, 'r') as f:
        pts_content = f.readlines()

    n_points_unparsed = pts_content[1]
    n_points = [int(s) for s in n_points_unparsed.split() if s.isdigit()][0]
    points = []

    assert len(pts_content) - 4 == n_points, '{} seems to have an incorrect number of lines'.format(file_path)

    for i, point_unparsed in enumerate(pts_content[3:-1]):
        points.append([float(p) for p in point_unparsed.split()])

    return {
        'n_points': n_points,
        'points': points
    }


def create_dataset_dict(args):
    root_dir = args._300w_path

    # Consolidate all information and place into coco format
    dataset_dict = {
        'categories': [],
        'images'    : []
    }
    img_count = 0

    for setting in SETTINGS:
        img_pts_path = os.path.join(root_dir, setting)
        img_pts_files = os.listdir(img_pts_path)

        img_paths = [f for f in img_pts_files if f.endswith('.png')]
        pts_paths = [f for f in img_pts_files if f.endswith('.pts')]

        pbar = tqdm(total=len(pts_paths))
        for f in pts_paths:
            pbar.update(1)

            img_file_name = os.path.splitext(f)[0] + '.png'
            img_path = os.path.join(img_pts_path, img_file_name)
            pts_path = os.path.join(img_pts_path, f)

            pts_data = parse_pts_file(pts_path)

            image_info = {
                'id'        : img_count,
                'file_name' : '/'.join([setting, img_file_name]),
                'bbox'      : None,
                'n_points'  : pts_data['n_points'],
                'landmarks' : pts_data['points']
            }
            image_info['width'], image_info['height'] = get_image_size(img_path)

            dataset_dict['images'].append(image_info)
            img_count += 1

        pbar.close()

    return dataset_dict


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script to preprocess 300W dataset into the coco format')

    # Most frequently used params
    parser.add_argument(metavar='300W_PATH', dest='_300w_path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str)

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    dataset_dict = create_dataset_dict(args)

    with open(os.path.join(args._300w_path, 'facial_landmark_full.json'), 'w') as f:
        json.dump(dataset_dict, f)

    print('Preprocessing done')


if __name__ == '__main__':
    main()
