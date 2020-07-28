"""
data_generator.py

Module to generate datasets based on requirements from J-lab

Author: CJ Paterno
Date: 02/20/2020
"""

from random import choice, randrange

import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.model_selection import train_test_split

from constants import normalizer, OPERATOR_IMAGES, OPERATORS, output_encoder

images, labels = loadlocal_mnist(
    images_path='data/train-images.idx3-ubyte',
    labels_path='data/train-labels.idx1-ubyte'
)

# Split data into train, val, test sets. Use random state to ensure the same elements are in
# the sets across training sessions
split_percent = 0.2
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=split_percent)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_percent)


def preprocess_input(some_input):
    """

    :param some_input:
    :return:
    """
    X = []

    for an_input in some_input:
        # Split from the input array, we know images are 28 x 28 and that there is a single
        # character for the operator between the two images
        a_image = [an_input[i] for i in range(0, 28 * 28)]
        a_image = np.reshape(a_image, (-1, 28)).astype(np.float)
        a_image = normalizer.transform(a_image)

        op = an_input[28 * 28]
        op_image = OPERATOR_IMAGES.get(op)

        b_image = [an_input[i] for i in range(28 * 28 + 1, len(an_input))]
        b_image = np.reshape(b_image, (-1, 28)).astype(np.float)
        b_image = normalizer.transform(b_image)

        input_images = [a_image, op_image, b_image]

        X.append(input_images)

    return np.array(X)


def generate_images(batch_size=250000, gen_mode=0):
    """

    :param batch_size:
    :param gen_mode: Generation mode
                        0: train set
                        1: validation set
                        2: test set
                        3: raw set
    :return: data, labels, operations
    """

    image_set = X_train
    label_set = y_train

    if gen_mode == 1:
        image_set = X_val
        label_set = y_val
    elif gen_mode == 2:
        image_set = X_test
        label_set = y_test
    elif gen_mode == 3:
        image_set = images
        label_set = labels

    X = []
    y = []
    operations = []

    for _ in range(batch_size):
        a_index = randrange(0, len(image_set))
        a_image = np.array(image_set[a_index]).reshape((-1, 28))
        a_image = normalizer.transform(a_image)
        a_label = label_set[a_index]

        op = choice(OPERATORS)
        op_image = np.array(OPERATOR_IMAGES.get(op))

        b_index = randrange(0, len(image_set))
        b_image = np.array(image_set[b_index]).reshape((-1, 28))
        b_image = normalizer.transform(b_image)
        b_label = label_set[b_index]

        X_input = [a_image, op_image, b_image]
        operation = f'{a_label} {op} {b_label}'
        y_input = [eval(operation)]
        operations.append(operation)

        X.append(X_input)
        y.append(y_input)

    X = np.array(X)
    y = output_encoder.transform(y).toarray()

    return X, y, operations


def generator(batch_size, gen_mode=0):
    """

    :param gen_mode:
    :param batch_size:
    :return:
    """

    while True:

        X, y, _ = generate_images(batch_size, gen_mode)

        yield X, y


def generate_example_input(num_examples=10000):
    """

    :param num_examples:
    :return:
    """
    X = []
    y = []
    operations = []

    for _ in range(num_examples):
        a_index = randrange(0, len(images))
        a_image = list(images[a_index])
        a_label = labels[a_index]

        op = choice(OPERATORS)

        b_index = randrange(0, len(images))
        b_image = list(images[b_index])
        b_label = labels[b_index]

        X_input = a_image
        X_input.extend(op)
        X_input.extend(b_image)

        operation = f'{a_label} {op} {b_label}'
        y_input = [eval(operation)]
        operations.append(operation)

        X.append(X_input)
        y.append(y_input)

    return X, y, operations
