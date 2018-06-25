import os
import sys
import json
import random
import argparse


SETS_TO_GENERATE = ['train2017', 'val2017']


def save_dataset_info(dataset_info, set_name, args):
    print('Saving {}'.format(set_name))

    with open(os.path.join(
        args.coco_path,
        'annotations',
        'instances_nocrowd_{}.json'.format(set_name)
    ), 'w') as f:
        json.dump(dataset_info, f)

    return dataset_info


def filter_dataset(dataset_info, set_name, args):
    print('Filtering {}'.format(set_name))

    # Filter crowds
    dataset_info['annotations'] = [ann for ann in dataset_info['annotations'] if ann['iscrowd'] == 0]

    return dataset_info


def load_dataset(set_name, args):
    print('Loading {}'.format(set_name))

    with open(os.path.join(
        args.coco_path,
        'annotations',
        'instances_{}.json'.format(set_name)
    ), 'r') as f:
        dataset_info = json.load(f)

    return dataset_info


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script to generate a batch of annotations for no crowd detection')

    # COCO path
    parser.add_argument('--coco-path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str, required=True)

    return parser.parse_args(args)


def main():
    # Set up script options
    args = parse_args(sys.argv[1:])

    for set_name in SETS_TO_GENERATE:
        dataset_info = load_dataset(set_name, args)
        filtered_dataset_info = filter_dataset(dataset_info, set_name, args)
        save_dataset_info(filtered_dataset_info, set_name, args)

    print('Done')


if __name__ == '__main__':
    main()
