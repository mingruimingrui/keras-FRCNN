import os
import sys
import json
import random
import argparse
from copy import deepcopy


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script to preprocess 300W dataset into the coco format')

    # Most frequently used params
    parser.add_argument(metavar='300W_PATH', dest='_300w_path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str)

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    with open(os.path.join(args._300w_path, 'facial_landmark_full.json'), 'r') as f:
        dataset_dict = json.load(f)

    img_ids = [img_info['id'] for img_info in dataset_dict['images']]
    n_imgs  = len(img_ids)
    test_ids = {id: None for id in random.sample(img_ids, int(n_imgs * 0.05))}

    train_imgs = [img_info for img_info in dataset_dict['images'] if img_info['id'] not in test_ids]
    test_imgs =  [img_info for img_info in dataset_dict['images'] if img_info['id']     in test_ids]

    train_dataset = deepcopy(dataset_dict)
    test_dataset  = deepcopy(dataset_dict)

    train_dataset['images'] = train_imgs
    test_dataset['images']  = test_imgs

    with open(os.path.join(args._300w_path, 'facial_landmark_train.json'), 'w') as f:
        json.dump(train_dataset, f)

    with open(os.path.join(args._300w_path, 'facial_landmark_test.json'), 'w') as f:
        json.dump(test_dataset, f)


if __name__ == '__main__':
    main()
