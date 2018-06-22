""" Script used to process the coco dataset annotations into a classification task

Please first arrange your coco dataset in this order under your coco root directory

coco_root_dir
  |- images
  |  |- train2017
  |  |- val2017
  |- annotations
  |  |- instances_train2017.json
  |  |- instances_val2017.json

"""

import os
import sys
import json
import argparse
from copy import deepcopy
from tqdm import tqdm


ANNOTATION_FILES = ['instances_train2017.json', 'instances_val2017.json']
IMAGE_SETS = ['train2017', 'val2017']


def save_dataset(dataset, img_set, args):
    print('Saving dataset to file')
    with open(os.path.join(
        args.coco_path,
        'annotations',
        'classification_{}.json'.format(img_set)
    ), 'w') as f:
        json.dump(dataset, f)


def create_classification_dataset(dataset, img_set):
    # Create empty list to store classification image_info
    class_imgs = []

    # Create an image dict for O(1) retrieval by image_id
    img_dict = {}
    for img in dataset['images']:
        img_dict[img['id']] = img

    pbar = tqdm(total=len(dataset['annotations']), desc='Processing {}'.format(img_set))

    for ann in dataset['annotations']:
        pbar.update(1)

        # Get copy of base image info
        img_info = deepcopy(img_dict[ann['image_id']])

        # Change id, append bbox and category_id
        img_info['id']           = ann['id']
        img_info['category_ids'] = ann['category_id']
        img_info['bboxes']       = ann['bbox']

        # Store into classification image list
        class_imgs.append(img_info)

    pbar.close()

    # Store all info and store as the classification dataset
    class_dataset = {
        'categories': dataset['categories'],
        'images': class_imgs
    }

    return class_dataset


def load_coco_dataset(ann_file, args):
    print('Now loading {}'.format(ann_file))
    with open(os.path.join(args.coco_path, 'annotations', ann_file), 'r') as f:
        dataset = json.load(f)
    return dataset


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script to preprocess coco annotations into a classification task')

    # Most frequently used params
    parser.add_argument(metavar='COCO_PATH', dest='coco_path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str)

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    for ann_file, img_set in zip(ANNOTATION_FILES, IMAGE_SETS):
        dataset = load_coco_dataset(ann_file, args)
        class_dataset = create_classification_dataset(dataset, img_set)
        save_dataset(class_dataset, img_set, args)

    print('Preprocessing done')


if __name__ == '__main__':
    main()
