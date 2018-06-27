""" Script used to process the kitti dataset annotations into one that is accepted by pycocotools
The objective here is to arrange into the format required by keras_pipeline.datasets.coco.DetectionDataset

Please first arrange your kitti dataset in this order under your kitti root directory

kitti_root_dir
  |- training
  |  |- image_2
  |  |- label_2
  |- testing (optional)
  |  |- image_2

If you are running windows OS, please remember to run this in administrator mode to enable symlink creation

"""

import os
import sys
import json
import argparse
from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

from keras_pipeline.preprocessing.image import get_image_size


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_kitti_annotation(text, image_id, ann_id, kitti_cat_to_id_dict):
    """ Parses the annotations from a text and returns in a dictionary form

    Refer to https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    for notes on annotation formats

    """
    ann_components = text.replace('\n', '').split(' ')

    # Ensure that annotation is valid else skip
    if not ann_components[0] in kitti_cat_to_id_dict:
        return None

    ann = {
        'id'          : ann_id,
        'image_id'    : image_id,
        'category_id' : kitti_cat_to_id_dict[ann_components[0]],
        'truncated'   : float(ann_components[1]),
        'occluded'    : int(ann_components[2]),
        'alpha'       : float(ann_components[3]),
        'rotation_y'  : float(ann_components[14]),
        'bbox'        : [float(x) for x in ann_components[4:8]],
        # 'iscrowd',
        # 'segmentation',
    }
    ann['bbox'][2] = ann['bbox'][2] - ann['bbox'][0]
    ann['bbox'][3] = ann['bbox'][3] - ann['bbox'][1]
    ann['area'] = ann['bbox'][2] * ann['bbox'][3]

    return ann


def get_kitti_info():
    return {
        'contributor': 'Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago',
        'description': 'KITTI Dataset for 2D object detection',
        'url': 'http://www.cvlibs.net/datasets/kitti/',
        'version': '1.0',
        'year': 2012
    }


def get_kitti_licenses():
    return [
        {'id': 1,
         'name': 'Attribution-NonCommercial-ShareAlike License',
         'url': 'http://creativecommons.org/licenses/by-nc-sa/3.0/'}
    ]


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


def get_kitti_images(set_name, args):
    set_image_path = os.path.join(args.kitti_path, set_name, 'image_2')
    image_file_names = os.listdir(set_image_path)
    image_infos = []

    pbar = tqdm(total=len(image_file_names), desc='Collecting image_infos')

    for i, img_fn in enumerate(image_file_names):
        pbar.update(1)

        # Get full image path
        full_img_path = os.path.join(set_image_path, img_fn)

        # Get all required image info
        img_info = {
            'id'        : int(os.path.splitext(img_fn)[0]),
            'file_name' : img_fn,
            'license'   : 1,
        }
        img_info['width'], img_info['height'] = get_image_size(full_img_path)

        # Append to image_infos list
        image_infos.append(img_info)

    pbar.close()

    return image_infos


def get_kitti_annotations(set_name, kitti_cat_to_id_dict, args):
    set_annotation_path = os.path.join(args.kitti_path, set_name, 'label_2')
    annotation_file_names = os.listdir(set_annotation_path)
    annotation_infos = []
    ann_id = 0

    pbar = tqdm(total=len(annotation_file_names), desc='Collecting annotation_infos')

    for i, ann_fn in enumerate(annotation_file_names):
        pbar.update(1)

        # Get image_id and full image annotation path
        image_id      = int(os.path.splitext(ann_fn)[0])
        full_ann_path = os.path.join(set_annotation_path, ann_fn)

        # Get all required detections from file
        with open(full_ann_path, 'r') as f:
            detections = f.readlines()

        # Parse detection into ann info
        ann_info = []
        for d in detections:
            ann = parse_kitti_annotation(d, image_id, ann_id, kitti_cat_to_id_dict)
            if ann is not None:
                ann_info.append(ann)
                ann_id += 1

        # Apeend to annotaion_infos
        annotation_infos += ann_info

    pbar.close()

    return annotation_infos


def make_kitti_cat_to_id_dict(categories):
    return { cat['name']: cat['id'] for cat in categories }


def create_image_anntation_dirs(kitti_path):
    annotations_dir = os.path.join(kitti_path, 'annotations')
    images_dir      = os.path.join(kitti_path, 'images')
    makedirs(annotations_dir)
    makedirs(images_dir)

    return annotations_dir, images_dir


def validate_kitti_dir(kitti_path):
    """ Checks that the kitti root directory is property arranged
    Also checks for the sets available to be preprocessed

    Args
        kitti_path : root directory of your kitti dataset

    Returns
        A list of set_names eg. training, testing
    """
    assert os.path.isdir(kitti_path), 'kitti_path provided is not a path'

    set_names = os.listdir(kitti_path)
    assert 'training' in set_names, 'training set must be in {}'.format(kitti_path)

    if 'testing' in set_names:
        set_names = ['training', 'testing']
    else:
        set_names = ['training']

    for set_name in set_names:
        set_path  = os.path.join(kitti_path, set_name)
        set_files = os.listdir(set_path)
        assert 'image_2' in set_files, 'the folder image_2 must be present in {}'.format(set_path)
        if set_name == 'training':
            assert 'label_2' in set_files, 'the folder label_2 must be present in {}'.format(set_path)

    return set_names


def check_args(args):
    args.set_names = validate_kitti_dir(args.kitti_path)
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

    # Create the annotations and images directories
    annotations_dir, images_dir = create_image_anntation_dirs(args.kitti_path)

    for set_name in args.set_names:
        # Format kitti annotations to the form required by coco
        formatted_kitti_annotations = {
            'info'        : get_kitti_info(),
            'licenses'    : get_kitti_licenses(),
            'categories'  : get_kitti_categories(args),
            'images'      : get_kitti_images(set_name, args),
            'annotations' : [],
        }

        # Get annotations if training set
        if set_name == 'training':
            kitti_cat_to_id_dict = make_kitti_cat_to_id_dict(formatted_kitti_annotations['categories'])
            formatted_kitti_annotations['annotations'] = get_kitti_annotations(set_name, kitti_cat_to_id_dict, args)

        # Save to json
        with open(os.path.join(annotations_dir, 'instances_{}.json'.format(set_name)), 'w') as f:
            json.dump(formatted_kitti_annotations, f)

        # Make symlinks
        os.symlink(
            os.path.join(args.kitti_path, set_name, 'image_2'),
            os.path.join(images_dir, set_name),
            True
        )


if __name__ == '__main__':
    main()
