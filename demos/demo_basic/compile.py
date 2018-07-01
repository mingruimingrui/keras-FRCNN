import os
import sys
import argparse

import keras

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

# Model
from keras_pipeline.models import RetinaNetConfig
from keras_pipeline.models import (
    RetinaNetFromTrain,
    LoadRetinaNet
)

# Dataset and generator
from keras_pipeline.datasets import DetectionDataset
from keras_pipeline.generators import DetectionGeneratorConfig, DetectionGenerator

# Evaluation
from keras_pipeline.utils.eval import evaluate_detection


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def eval_model(generator, model, args):
    makedirs(args.eval_dir)
    APs = evaluate_detection(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        max_images=None,
        max_plots=args.max_plots,
        save_path=args.eval_dir
    )

    import json
    with open('temp.json', 'w') as f:
        json.dump(APs, f)


def make_generators(validation_set, model_config, args):
    validation_generator_config = DetectionGeneratorConfig(
        dataset = validation_set,
        model_config = model_config,
        batch_size = 1
    )

    validation_generator = DetectionGenerator(validation_generator_config)

    return validation_generator


def save_prediction_model(model, args):
    makedirs(args.save_dir)
    model.save(os.path.join(args.save_dir, args.save_name))


def load_prediction_model(model_config, args):
    training_model = LoadRetinaNet(args.snapshot, model_config.backbone_name)
    prediction_model = RetinaNetFromTrain(training_model, model_config)

    return prediction_model


def load_validation_set(args):
    # Load dataset information
    validation_set = DetectionDataset(args.coco_path, 'instances_val2017.json'  , 'val2017'  )

    return validation_set


def parse_args(args):
    parser = argparse.ArgumentParser(description='Demo compile script for compiling training snapshots into prediction models.')

    # Most frequently used params
    parser.add_argument(metavar='COCO_PATH', dest='coco_path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str)
    parser.add_argument(metavar='SNAPSHOT', dest='snapshot',
        help='Training snapshot file to load model weights from',
        type=str)
    parser.add_argument('--save-dir',
        help='Save path for compiled prediction model',
        default='./prediction_model')
    parser.add_argument('--save-name',
        help='Name of saved prediction model',
        default='retinanet_resnet50_coco_pred.h5')
    parser.add_argument('--eval',
        help='Perform evaluation on a fixed number of images',
        dest='evaluation', action='store_true')
    parser.add_argument('--eval-dir',
        help='Save path for the evaluated images and report',
        default='./eval')
    parser.add_argument('--max-plots',
        help='Maximum number of images to plot if evaluating',
        default=10)

    return parser.parse_args(args)


def main():
    # Set up script options
    args = parse_args(sys.argv[1:])

    # Load dataset information
    validation_set = load_validation_set(args)

    # Load model
    print('\n==== Loading Model ====')
    print('This can take a while...')
    model_config = RetinaNetConfig(num_classes = validation_set.get_num_object_classes())
    model = load_prediction_model(model_config, args)
    print('Model created, now saving')
    save_prediction_model(model, args)
    print('Prediction model saved')

    # Exit if not evaluating
    if not args.evaluation:
        return

    # Make the validation set generator
    # The reason why we create model first is because we need to know
    # how to create anchors and preprocess image (based on backbone)
    print('\n==== Making Data Generator ====')
    print('This can take a while...')
    validation_generator = make_generators(
        validation_set,
        model_config = model_config,
        args = args
    )
    print('Data Generator created')

    print('\n==== Evaluating Model ====')
    print('This can take a while...')
    eval_model(validation_generator, model, args)
    print('Evaluating complete')


if __name__ == '__main__':
    print('\n==== Starting compile.py ====')
    main()
    print('\n==== Finished compile.py ====')
