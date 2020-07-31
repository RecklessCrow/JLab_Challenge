"""
generate_dataset_to_file.py

Main interface to solve the ML Lunch Group problem number 5

Author: CJ Paterno
Date: 02/20/2020
"""
import csv
import os

import keras
import numpy as np

from constants import output_encoder
from data_generator import preprocess_input
from train_model import get_results


# Score to beat, SSE: 35880.24
# Use CPU for training. Remove for GPU training.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    """
    Main function
    :return:
    """

    test_data_path = os.path.join('data', 'test_data.csv')
    test_label_path = os.path.join('data', 'test_labels.csv')

    with open(test_data_path) as f:
        reader = csv.reader(f)
        X = list(reader)

    with open(test_label_path) as f:
        reader = csv.reader(f)
        y = np.array(list(reader)).astype(np.int)

    X = preprocess_input(X)

    model_path = os.path.join('checkpoints', '2020-07-30_15-26')
    model = keras.models.load_model(model_path)

    predictions = output_encoder.inverse_transform(model.predict(X))

    get_results(predictions, y)


if __name__ == '__main__':
    main()
