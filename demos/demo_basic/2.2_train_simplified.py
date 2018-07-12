import os
import sys
import argparse

from dataset_pipeline import ImageDataset

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

from keras_pipeline.utils.logging import set_default_logging

from keras_pipeline.models import RetinaNetConfig
from keras_pipeline.generators import DetectionGeneratorConfig

from keras_pipeline.training import train_model


def determine_initial_epoch(args):
    if args.snapshot:
        initial_epoch = os.path.splitext(args.snapshot)[0].split('_')[-1]
        return int(initial_epoch)
    return 0


def check_args(args):
    assert args.num_gpu >= 1, 'Must train with atleast 1 GPU'
    assert args.batch_size >= args.num_gpu, 'Batch size must be equal or greater than number of GPUs used'

    return args


def parse_args(args):
    parser = argparse.ArgumentParser(description='Demo training script for training a RetinaNet network.')

    # Most frequently used params
    parser.add_argument(metavar='COCO_PATH', dest='coco_path',
        help='Path to dataset directory (ie. /tmp/COCO)',
        type=str)
    parser.add_argument('--num-gpu',
        help='Number of gpus to train model with, you must train with atleast 1 GPU',
        default=1, type=int)
    parser.add_argument('--batch-size',
        help='Size of the batches',
        default=1, type=int)

    # Resume training / load weights
    parser.add_argument('--snapshot',
        help='Resume training from a snapshot file')
    parser.add_argument('--snapshot-path',
        help='Path to store snapshots of model during training',
        default='./snapshot')
    parser.add_argument('--no-snapshots',
        help='Disable saving snapshots overrides snapshot path',
        dest='snapshots', action='store_false')

    # Logging params
    parser.add_argument('--log-dir',
        help='Log directory for misc outputs and log file',
        default='./logs')
    parser.add_argument('--tensorboard-dir',
        help='Log directory for Tensorboard output',
        default='./logs')
    parser.add_argument('--eval',
        help='Perform evaluation per epoch',
        dest='evaluation', action='store_true')

    return parser.parse_args(args)


def get_args(args):
    return check_args(parse_args(args))


def main():
    # Set up script options
    args = get_args(sys.argv[1:])

    # Config logger
    set_default_logging(os.path.join(args.log_dir, 'main.log'))

    # Load dataset API
    train_set = ImageDataset(os.path.join(args.coco_path, 'annotations', 'pipeline_train2017.json'))
    val_set   = ImageDataset(os.path.join(args.coco_path, 'annotations', 'pipeline_val2017.json'))

    # Create model and generator config files
    model_config = RetinaNetConfig(num_classes=train_set.get_num_classes())
    train_generator_config = DetectionGeneratorConfig(
        dataset=train_set,
        model_config=model_config,
        batch_size = args.batch_size,
        allow_transform = True,
        shuffle_groups = True
    )
    val_generator_config   = DetectionGeneratorConfig(
        dataset = val_set,
        model_config = model_config,
        batch_size = args.batch_size
    )

    # Train model
    train_model(
        model_config,
        train_generator_config,
        val_generator_config=val_generator_config,
        num_gpu=args.num_gpu
        epochs=50,
        steps_per_epoch=10000,
        log_dir=args.log_dir,
        tensorboard_dir=args.tensorboard_dir,
        snapshot_path=args.snapshot_path,
        snapshot_name='retinanet_coco_{epoch:02d}.h5',
        evaluation=args.evaluation,
        snapshot=args.snapshot,
        initial_epoch=determine_initial_epoch(args)
    )


if __name__ == '__main__':
    main()
