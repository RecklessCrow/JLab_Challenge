"""
train_model.py

Module to create, train, and test a keras model

Author: CJ Paterno
Date: 02/20/2020
"""
import os
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential

from constants import output_encoder
from data_generator import generator, generate_images

# Use CPU for training. Remove for GPU training.
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Fix memory allocation issues for GPU training
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def make_model(num_out=49):
    """
    Make a new keras model
    :param num_out: number of outputs
    :return: a keras model
    """

    model = Sequential()

    model.add(Permute(
        (2, 3, 1),
        input_shape=(3, 28, 28)
    ))

    model.add(Conv2D(
        32,
        (8, 8),
        strides=(4, 4),
        activation='relu',
    ))

    model.add(Conv2D(
        64,
        (4, 4),
        strides=(2, 2),
        activation='relu'
    ))

    model.add(Conv2D(
        64,
        (2, 2),
        strides=(1, 1),
        activation='relu'
    ))

    model.add(Flatten())

    model.add(Dense(
        units=512,
        activation='relu'
    ))

    model.add(Dense(
        units=num_out,
        activation='softmax'
    ))

    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'mse']
    )

    return model


# Current date to the minute for logging
curr_date = datetime.now().strftime("%Y-%m-%d_%H-%M")


def train(load_file=None, save_file=None):
    """
    Train a keras model
    :param load_file: load a previously saved checkpoint or model
    :param save_file: file to save trained model to
    """
    epochs = 50
    steps_per_epoch = 1000
    batch_size = 5000

    # Create model
    # Load checkpoint if exists
    if load_file is not None:
        model = keras.models.load_model(load_file)
    else:
        model = make_model()

    print(model.summary())

    # Create callbacks
    log_dir = os.path.join('logs', 'scalars', curr_date)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_file = os.path.join('checkpoints', curr_date)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, monitor='val_loss',
                                                          verbose=1, save_best_only=True, mode='min')

    # Train model
    model.fit(
        generator(batch_size=batch_size, gen_mode=0),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=generator(batch_size=batch_size // 3, gen_mode=1),
        validation_steps=steps_per_epoch // 3,
        callbacks=[tensorboard_callback, checkpoint_callback],
    )

    score = test(model)

    if save_file is None:
        model.save(os.path.join('models', str(score)))
    else:
        model.save(save_file)


def test(model):
    """
    Test a model
    :param model: Model to test
    :return: Sum squared error
    """

    print('\nTesting...')

    x, y, operations = generate_images(batch_size=500000, gen_mode=2)
    predictions = model.predict(x)
    predictions = output_encoder.inverse_transform(predictions)
    y = output_encoder.inverse_transform(y)

    sse = get_results(predictions, y, operations)

    return sse


def get_results(actual, expected, operations=None):
    """
    Calculate sum squared error and print test results
    :param actual: Model predicted results
    :param expected: Expected results
    :param operations: List of operations preformed to get results
    :return: Sum squared error
    """

    sse = np.sum(np.square(actual - expected))
    num_wrong = 0

    for i, result in enumerate(actual):
        if result != expected[i][0]:
            if operations is not None:
                print(f'Expected: {operations[i]} = {expected[i][0]}\n'
                      f'Actual:   {result[0]}\n')
            num_wrong += 1

    percent_wrong = num_wrong / len(expected) * 100

    print(f'Results\n'
          f'\tTotal Wrong   - {num_wrong} / {len(expected)}\n'
          f'\tAccuracy      - {100 - percent_wrong:.2f}%\n'
          f'\tSSE           - {sse}')

    return sse


if __name__ == '__main__':
    checkpoint = os.path.join('checkpoints', '2020-07-30_00-31')
    train(save_file=checkpoint)
