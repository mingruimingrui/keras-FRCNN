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
        'person_instances_{}.json'.format(set_name)
    ), 'w') as f:
        json.dump(dataset_info, f)

    return dataset_info


def filter_dataset(dataset_info, set_name, args):
    print('Filtering {}'.format(set_name))

    # Filter out person and person_crowd annotations
    person_annotations = [ann for ann in dataset_info['annotations']
                          if (ann['category_id'] == 1) & (ann['iscrowd'] == 0)]
    crowd_annotations  = [ann for ann in dataset_info['annotations']
                          if (ann['category_id'] == 1) & (ann['iscrowd'] == 1)]

    # Get associated image_ids to annotations containing person and person_crowd
    all_image_ids    = [image_info['id'] for image_info in dataset_info['images']]
    person_image_ids = [ann['image_id'] for ann in person_annotations]
    crow_image_ids   = [ann['image_id'] for ann in crowd_annotations]

    image_id_to_keep = []
    if args.allow_negative:
        # If allow negative, then we filter out some negative images to add to the dataset
        image_id_to_keep = [id for id in all_image_ids
                            if (id not in person_image_ids) & (id not in crow_image_ids)]
        image_id_to_keep = random.sample(image_id_to_keep, args.limit_negative)

    if args.allow_crowd:
        # Get the image_id of the images which contain person annotations
        image_id_to_keep  += [ann['image_id'] for ann in person_annotations]
    else:
        # Get the image_id of the images which contain person annotations but not person_crowd_annotations
        image_id_to_ignore = [ann['image_id'] for ann in crowd_annotations]
        image_id_to_ignore = {k: None for k in image_id_to_ignore}
        image_id_to_keep  += [ann['image_id'] for ann in person_annotations
                              if ann['image_id'] not in image_id_to_ignore]

    # Convert image_id_to_keep into dict for O(1) lookup
    image_id_to_keep = {k: None for k in image_id_to_keep}

    # Now consolidate the new dataset_info
    new_info        = dataset_info['info']
    new_licenses    = dataset_info['licenses']
    new_categories  = dataset_info['categories'][:1]
    new_images      = [image_info for image_info in dataset_info['images'] if image_info['id'] in image_id_to_keep]
    new_annotations = [ann for ann in person_annotations if ann['image_id'] in image_id_to_keep]

    new_dataset_info = {
        'info'        : new_info,
        'licenses'    : new_licenses,
        'categories'  : new_categories,
        'images'      : new_images,
        'annotations' : new_annotations,
    }

    return new_dataset_info


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
    parser = argparse.ArgumentParser(description='Script to generate a batch of annotations for person detection')

    # COCO path
    parser.add_argument('--coco-path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str, required=True)

    # Allow crowd
    parser.add_argument('--allow-crowd',
        help='Allow annotations to include person_crowd',
        action='store_true')

    # Allow negative images
    parser.add_argument('--allow-negative',
        help='Allow some images with no person to be included',
        action='store_true')
    parser.add_argument('--limit-negative',
        help='Limit on number of negative images',
        type=int, default=300)

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
