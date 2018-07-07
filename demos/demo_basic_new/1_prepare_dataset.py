import os
import sys
import time
import argparse
import json

from dataset_pipeline import ImageDataset


# Constant variables
DATASETS_TO_PREPARE = ['train2017', 'val2017']


def transform_dataset(dataset, set_name, args):
    """ Transforms a coco dataset into a pipeline dataset """
    # Create new dataset object to store dataset information
    pipeline_dataset = ImageDataset(args.outfile.format(set_name))

    # Set all image information
    root_dir    = os.path.join(args.coco_path, 'images')
    image_paths = [os.path.join(set_name, img['file_name']) for img in dataset['images']]
    image_urls  = [img['coco_url'] for img in dataset['images']]
    image_ids   = [img['id'] for img in dataset['images']]
    pipeline_dataset.create_dataset(
        root_dir=root_dir,
        image_paths=image_paths,
        image_urls=image_urls,
        image_ids=image_ids,
        force_overwrite=True
    )

    # Set classes
    class_names = [cat['name'] for cat in dataset['categories']]
    pipeline_dataset.set_classes(class_names)

    # Store annotations individually
    for ann in dataset['annotations']:
        bbox = ann['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        pipeline_dataset.set_single_image_annotation(
            image_id=ann['image_id'],
            bbox=bbox,
            ann_class=ann['category_id'],
            segmentation=ann['segmentation']
        )

    return pipeline_dataset


def load_coco_dataset(set_name, args):
    with open(os.path.join(args.coco_path, 'annotations', 'instances_{}.json'.format(set_name)), 'r') as f:
        coco_data = json.load(f)
    return coco_data


def parse_args(args):
    parser = argparse.ArgumentParser(description='Demo dataset preparation script.')

    parser.add_argument(metavar='COCO_PATH', dest='coco_path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str)
    parser.add_argument('-O', '--outfile',
        help='File to store dataset information as a json file, default is ',
        type=str)

    return parser.parse_args(args)


def check_args(args):

    # Parse outfile as just the base file name
    if args.outfile is None:
        args.outfile = os.path.join(
            os.path.abspath(args.coco_path),
            'annotations',
            'pipeline_{}.json')
    else:
        args.outfile = os.path.splitext(args.outfile)[0] + '_{}.json'

    return args


def get_args(args):
    return check_args(parse_args(args))


def main():
    args = get_args(sys.argv[1:])

    for dataset_name in DATASETS_TO_PREPARE:
        t = time.time()
        dataset = load_coco_dataset(dataset_name, args)
        pipeline_dataset = transform_dataset(dataset, dataset_name, args)
        print('{} preprocessed in {}s'.format(dataset_name, time.time() - t))
        pipeline_dataset.save_dataset()


if __name__ == '__main__':
    main()
