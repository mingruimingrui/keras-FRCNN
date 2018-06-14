""" File used to split and/or filter a coco format annotation file
This file is written specifically for the kitti dataset """

import os
import sys
import random
import json
import argparse
from tqdm import tqdm


def save_dataset(kitti_path, dataset, file_name):
    with open(os.path.join(kitti_path, 'annotations', file_name), 'w') as f:
        json.dump(dataset, f)


def split_train_val(annotation_data, args):
    # Get img ids for validation set
    all_img_ids = [img['id'] for img in annotation_data['images']]
    num_val_img = int(args.val_ratio * len(all_img_ids))
    val_img_ids = random.sample(all_img_ids, num_val_img)

    # Transform val_img_ids into dict for O(1) lookup
    val_img_ids = {k: None for k in val_img_ids}

    # Get training and validation images
    training_images   = []
    validation_images = []
    for img in annotation_data['images']:
        if img['id'] in val_img_ids:
            validation_images.append(img)
        else:
            training_images.append(img)

    # Get training and validation annotations
    training_annotations   = []
    validation_annotations = []
    for ann in annotation_data['annotations']:
        if ann['image_id'] in val_img_ids:
            validation_annotations.append(ann)
        else:
            training_annotations.append(ann)

    if args.use_full_train:
        training_images      = annotation_data['images']
        training_annotations = annotation_data['annotations']

    # Consolidate training and validation data
    training_data = {
        'info'        : annotation_data['info'],
        'licenses'    : annotation_data['licenses'],
        'categories'  : annotation_data['categories'],
        'images'      : training_images,
        'annotations' : training_annotations,
    }

    validation_data = {
        'info'        : annotation_data['info'],
        'licenses'    : annotation_data['licenses'],
        'categories'  : annotation_data['categories'],
        'images'      : validation_images,
        'annotations' : validation_annotations,
    }

    return training_data, validation_data


def filter_annotations(annotation_data, args):
    """ Filters the annotations based on the user requirements
    Removes images with no annotations

    Returns the filtered annotation data
    """
    filtered_annotations = []
    filtered_images      = []
    img_id_keep        = []

    # Filter annotations
    for ann in annotation_data['annotations']:
        if (ann['truncated'] <= args.truncated_threshold) and (ann['occluded'] <= args.occluded_threshold):
            filtered_annotations.append(ann)
            img_id_keep.append(ann['image_id'])
        else:
            print('wtf')

    # Transform img_id_keep into dict for O(1) lookup
    img_id_keep = {k: None for k in img_id_keep}

    # filter images based on remaining annotations
    for img in annotation_data['images']:
        if img['id'] in img_id_keep:
            filtered_images.append(img)
        else:
            print('WTF')

    # Update annotation_data with the filtered annotations and images
    annotation_data['annotations'] = filtered_annotations
    annotation_data['images']      = filtered_images

    return annotation_data


def get_full_annotation_data(kitti_path):
    with open(os.path.join(kitti_path, 'annotations', 'instances_training.json'), 'r') as f:
        annotation_data = json.load(f)
    return annotation_data



def parse_args(args):
    parser = argparse.ArgumentParser(description='Script to preprocess kitti annotations and directories to replicate COCO required formats')

    # Most frequently used params
    parser.add_argument('--kitti-path',
        help='Path to dataset_directory (ie. tmp/KITTI)',
        type=str, required=True)

    # Method to split dataset
    parser.add_argument('--val-ratio',
        help='Portions this percent of images from entire image set to validation set',
        type=float, default=0.05)
    parser.add_argument('--use-full-train',
        help='Use the entire image set for training',
        action='store_true')

    # Dataset configs
    parser.add_argument('--max-truncated',
        help='Maximum truncated threshold between 0 (not truncated) and 1 (very truncated)',
        type=float, default=1., dest='truncated_threshold')
    parser.add_argument('--max-occluded',
        help='Maximum level of occlusion allowed between 0 (fully visible) and 3 (largely occluded)', # I know that 2 is largely occluded
        type=int, default=3, dest='occluded_threshold')

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    # Load the entire image set info
    print('Loading full dataset')
    full_annotation_data = get_full_annotation_data(args.kitti_path)

    # Filter annotations
    print('Filtering unnecessary annotations')
    full_annotation_data = filter_annotations(full_annotation_data, args)

    # Split into train and val set
    print('Spliting dataset into train and val')
    training_set, validation_set = split_train_val(full_annotation_data, args)

    # Save train and val set
    save_dataset(args.kitti_path, training_set  , 'instances_local_train.json')
    save_dataset(args.kitti_path, validation_set, 'instances_local_val.json')

    print('Dataset filtered, split and saved')


if __name__ == '__main__':
    main()
