# Non Coco Demo

This experiment is to showcase how one can leverage off pycocotools to fit a
dataset to fit the input requirements as specified in CocoDataset
(from keras_pipeline/datasets/coco.py). Here we specifically will demonstrate
with the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).


## Table of content / steps to take

1. [Create annotation file and symlinks](#create-annotation-file-and-symlinks)
2. [Split training images](#split-training-images)
3. [Train and evaluate](#train-and-evaluate)


## Create annotation file and symlinks

The coco dataset require your directory to be arranged as described in the following
tree diagram.

```
root_dir
  |- images
  |  |- <set_1>
  |  |- <set_2>
  |  |- <set_3>
  |  |- ...
  |- annotations
```

The COCO annotation file will be in a JSON format with the following contents.

```
# Taking dataset_info as the root
>>> dataset_info.keys() # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

>>> dataset_info['info']
    # {'contributor': 'COCO Consortium',
    #  'date_created': '2017/09/01',
    #  'description': 'COCO 2017 Dataset',
    #  'url': 'http://cocodataset.org',
    #  'version': '1.0',
    #  'year': 2017}

>>> type(dataset_info['licenses']) # list
>>> dataset_info['licenses'][0]
    # {'id': 1,
    #  'name': 'Attribution-NonCommercial-ShareAlike License',
    #  'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'}

>>> type(dataset_info['categories']) # list
>>> dataset_info['categories'][0]
    # {'id': 1, 'name': 'person', 'supercategory': 'person'}

>>> type(dataset_info['images']) # list
>>> dataset_info['images'][0]
    # {'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
    #  'date_captured': '2013-11-14 17:02:52',
    #  'file_name': '000000397133.jpg',
    #  'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
    #  'height': 427,
    #  'id': 397133,
    #  'license': 4,
    #  'width': 640}

>>> type(dataset_info['annotations']) # list
>>> dataset_info['annotations'][0]
    # {'area': 702.1057499999998,
    #  'bbox': [473.07, 395.93, 38.65, 28.67],
    #  'category_id': 18,
    #  'id': 1768,
    #  'image_id': 289343,
    #  'iscrowd': 0,
    #  'segmentation' ...}
```

To use the kitti dataset with existing ```keras_pipeline``` functions, one can
create an annotation file from the kitti annotations to fit to the coco
annotation style. Next to replicate the required directories in the coco dataset,
one can utilize symlinks.

The script ```preprocess_dataset.py``` performs such a transformation.
To run it, use the following command at the directory ```non_coco_demo```.
(the directory of this README)

Inside is a script that parses the kitti annotations into a coco accepted
format and stores the data into an annotation file under
<ROOT_KITTI_DIR>/annotations. The file would also create an images directory
and create symlinks to the related image sets.

```
# admin rights are required on windows OS to create symlinks
python 1_preprocess_dataset.py --kitti-path <PATH TO YOUR KITTI DATASET>
```


## Split training images

The ```testing``` kitti dataset does not contain any labels so we have to
split their training set into a local training and validation set. A
script is written for you to run. Simply use

```
python 2_split_filter_annotaion.py --kitti-path <PATH TO YOUR KITT DATASET>
```

Detection models don't tend to overfit too much so it is often fine to
sample from the training set for evaluation purposes.

However since many of us were taught to always use unexplored data to do
our evaluation, the default in the file is to exclude validation data from
the training set.


## Train and evaluate

You can now write a training file and run your training task on your new dataset.
Here I'm borrowing ```train.py``` directly from ```basic_demo```. Simply I'm editting
out a few changes to reference to ```kitti_path``` instead of ```coco_path```.
(Also using the ```kitti set_names```). To run the training file, just use

```
python 3_train.py --kitti-path <PATH TO YOUR KITT DATASET>
```

Evaluation is still TBI.
