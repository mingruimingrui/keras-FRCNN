""" Scripts for training a model using keras """
import os
import json
import random
import logging

import tensorflow as tf
import keras


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_dict(dict_obj, file_path):
    # Identify items that annot be converted to json
    keys_to_delete = []
    for key, value in dict_obj.items():
        try:
            json.dumps(value)
        except TypeError:
            keys_to_delete.append(key)

    # Delete items
    if len(keys_to_delete) > 0:
        logging.warn('The objects {} cannot be converted to json'.format(keys_to_delete))
        for key in keys_to_delete:
            del dict_obj[key]

    # Save
    with open(file_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)


def config_session():
    session_config = tf.ConfigProto()

    # Allow growth
    session_config.gpu_options.allow_growth = True

    # Set config
    current_session = tf.Session(config=session_config)
    keras.backend.tensorflow_backend.set_session(current_session)


def log_training_config(args):
    if not args['log_dir']:
        logging.warn('logging directory not provided, no training byproducts will be created')
    if not args['tensorboard_dir']:
        logging.warn('tensorboard directory not provided, no tensorboard objects will be created')
    if not args['snapshot_path']:
        logging.warn('snapshot_path not found, no training snapshots will be saved')
    if args['snapshot']:
        logging.info('Previous model found, training will continue from {}'.format(args['snapshot']))

    logging.info('')
    logging.info('==================== General Info ====================')
    logging.info('model_name : {}'.format(args['model_name']))

    logging.info('')
    logging.info('==================== Logging Info ====================')
    if args['log_dir']:
        logging.info('log_dir         : {}'.format(args['log_dir']))
    if args['tensorboard_dir']:
        logging.info('tensorboard_dir : {}'.format(args['tensorboard_dir']))
    if args['snapshot_path']:
        logging.info('snapshot_path   : {}'.format(args['snapshot_path']))
    if (not args['log_dir']) and (not args['tensorboard_dir']) and (not args['snapshot_path']):
        logging.warn('No logs will be saved')

    logging.info('')
    logging.info('==================== Training Info ====================')
    logging.info('epochs          : {}'.format(args['epochs']))
    logging.info('steps_per_epoch : {}'.format(args['steps_per_epoch']))

    if args['log_dir']:
        makedirs(args['log_dir'])
        model_config_save_path = os.path.join(args['log_dir'], '{}_config.json'.format(args['model_name']))
        save_dict(args['model_config'].as_attr_dict(), model_config_save_path)
        logging.info('')
        logging.info('==================== Model Config Info ====================')
        logging.info('model_config_json_file : {}'.format(model_config_save_path))


def load_model(model_name, model_config, snapshot=None, evaluation=False):
    logging.info('')
    logging.info('==================== Making Model ====================')
    logging.info('This can take a while')

    if model_name == 'RetinaNet':
        from ..models.retinanet import RetinaNetTrain     as model_train
        from ..models.retinanet import RetinaNetFromTrain as model_from_train
        from ..models.retinanet import LoadRetinaNet      as model_from_snapshot
    else:
        raise Exception('So far only RetinaNet has been implemented')

    if snapshot is None:
        training_model = model_train(model_config)
    else:
        training_model = model_from_snapshot(snapshot, model_config.backbone_name)

    prediction_model = None
    if evaluation:
        prediction_model = model_from_train(training_model, model_config)

    logging.info('Model created')

    return training_model, prediction_model


def load_generators(train_generator_config, val_generator_config):
    generator_config_name = train_generator_config.__class__.__name__

    if generator_config_name == 'DetectionGeneratorConfig':
        from ..generators.detection import DetectionGenerator as generator
    elif generator_config_name == 'ImageClassGeneratorConfig':
        from ..generators.image_class import ImageClassGenerator as generator
    else:
        raise Exception('{} has not been implemented'.format(generator_config_name))

    train_generator = generator(train_generator_config)
    validation_generator = None
    if val_generator_config is not None:
        validation_generator = generator(val_generator_config)

    return train_generator, validation_generator


def log_sample_inputs(generator, log_dir=None):
    if not log_dir:
        return
    makedirs(log_dir)
    logging.info('')
    logging.info('==================== Saving sample inputs ====================')
    logging.info('This can take a while')

    # Get sample batch input
    group = [random.choice(generator.all_image_index)]
    batch = generator._get_batches_of_transformed_samples(group)

    # Get generator_name
    generator_name = generator.__class__.__name__

    # Save sample inputs
    if generator_name == 'DetectionGenerator':
        import numpy as np
        from ..utils.visualization import draw_annotations
        from ..utils.image_io import save_image

        # Gather batch data
        image       = batch[0][0].astype('uint8')
        labels      = batch[1][0][0]
        regressions = batch[1][1][0]

        # Compute bboxes
        anchors = generator.compute_anchors(image.shape)
        anchor_widths  = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        bboxes = regressions[:, :4]
        bboxes *= 0.2
        bboxes[:, 0] *= anchor_widths
        bboxes[:, 1] *= anchor_heights
        bboxes[:, 2] *= anchor_widths
        bboxes[:, 3] *= anchor_heights
        bboxes += anchors

        # Remove ignore
        select = regressions[:, 4] == 1
        labels = labels[select]
        bboxes = bboxes[select]

        # Draw annotations and save
        annotations = np.hstack([bboxes, labels])
        draw_annotations(image, annotations, label_to_name=generator.label_to_name)
        save_image(image, os.path.join(log_dir, 'sample_input.jpg'))

    else:
        logging.warn('{} has not been implemented for logging sample inputs'.format(generator_name))

    logging.info('Sample inputs saved')


def create_callback(
    model_name,
    training_model,
    batch_size,
    log_dir='./logs',
    tensorboard_dir='./logs',
    snapshot_path='./snapshot',
    snapshot_name='model_{epoch:02d}.h5',
    evaluation=False,
    prediction_model=None,
    validation_generator=None
):
    logging.info('')
    logging.info('==================== Making Callbacks ====================')
    logging.info('This can take a while')

    callbacks = []

    # Create loss logger
    if log_dir:
        from ..callbacks._misc import ProgressLogger
        makedirs(log_dir)
        train_progress_file = os.path.join(log_dir, 'train.log')
        progress_callback = ProgressLogger(log_path=train_progress_file, stdout=True)
        callbacks.append(progress_callback)

    # Create tensorboard
    tensorboard_callback = None
    if tensorboard_dir:
        makedirs(tensorboard_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = batch_size,
            write_graph            = True,
            write_grads            = True,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    # Save model
    if snapshot_path:
        makedirs(snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(snapshot_path, snapshot_name),
            verbose=1,
            save_weights_only=False,
        )
        callbacks.append(checkpoint)

    if evaluation and prediction_model and validation_generator:
        from ..callbacks import RedirectModel
        if model_name == 'RetinaNet':
            from ..callbacks import EvaluateDetection as evaluate_fn
        else:
            raise Exception('So far only RetinaNet has been implemented')
        evaluation = evaluate_fn(validation_generator, tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor   = 'loss',
        factor    = 0.1,
        patience  = 2,
        verbose   = 1,
        mode      = 'auto',
        min_delta = 1e-4,
        cooldown  = 0,
        min_lr    = 0
    ))

    logging.info('Callbacks created')

    return callbacks


def train_model(
    model_config,
    train_generator_config,
    job_name=None,
    epochs=50,
    steps_per_epoch=None,
    log_dir='./logs',
    tensorboard_dir='./logs',
    snapshot_path='./snapshot',
    snapshot_name='model_{epoch:02d}.h5',
    evaluation=False,
    val_generator_config=None,
    snapshot=None,
    initial_epoch=0,
    workers=1,
    use_multiprocessing=False
):
    """ Trains a model based on the model config and generator config, returns a trained model
    Args
        model_config           : A model config from keras_pipeline.models
        train_generator_config : A generator config from keras_pipeline.generators containing the training set
        job_name               : The name of this training job (used to mark the output files)

        epochs          : Number of epoch to train model on, default 50
        steps_per_epoch : Number of steps per epoch, default entire dataset

        log_dir         : Directory to save a log file (set to None to not output)
        tensorboard_dir : Directory to save a tensorboard file (set to None to not output)
        snapshot_path   : Directory to save model snapshots (set to None to not output)
        snapshot_name   : Name of snapshot files (follows https://keras.io/callbacks/#ModelCheckpoint)

        evaluation           : Flag to perform evaluation at end of each epoch (uses validation set to evaluate)
        val_generator_config : A generator config from keras_pipeline.generators containing the validation set (Not needed if not evaluating)

        snapshot      : h5 model file to resume training from
        initial_epoch : Epoch at which to start training

        verbose : Verbose level as defined in https://keras.io/models/model/#fit_generator

        workers             : Number of workers to generate training data with
        use_multiprocessing : Use process based threading

    Returns
        A trained keras model
    """
    logging.info('==================== Initializing Training Job ====================')

    config_session()

    # Get model_name
    model_name = model_config.__class__.__name__.replace('Config', '')

    # Log training configs
    log_training_config(args=locals())

    # Load models and generators from config objects
    training_model, prediction_model = load_model(model_name, model_config, snapshot, evaluation)
    train_generator, validation_generator = load_generators(train_generator_config, val_generator_config)

    # Generate sample of input images
    log_sample_inputs(train_generator, log_dir)

    # Create callbacks
    callbacks = create_callback(
        model_name,
        training_model,
        train_generator.batch_size,
        log_dir=log_dir,
        tensorboard_dir=tensorboard_dir,
        snapshot_path=snapshot_path,
        snapshot_name=snapshot_name,
        evaluation=evaluation,
        prediction_model=prediction_model,
        validation_generator=validation_generator
    )

    logging.info('')
    logging.info('==================== Training Start ====================')

    # Begin training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )

    logging.info('')
    logging.info('==================== Training Done ====================')
