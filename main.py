"""
generate_dataset_to_file.py

Main interface to solve the ML Lunch Group problem number 5

Author: CJ Paterno
Date: 02/20/2020
"""
import csv

import keras
import numpy as np

from constants import output_encoder
from data_generator import preprocess_input
from train_model import get_results


def main():
    """
    Main function
    :return:
    """

    with open('./data/test_data.csv') as f:
        reader = csv.reader(f)
        X = list(reader)

    with open('./data/test_labels.csv') as f:
        reader = csv.reader(f)
        y = np.array(list(reader)).astype(np.int)

    operations = []

    X = preprocess_input(X)

    model = keras.models.load_model('./models/2404574')

    print(model.summary())

    predictions = output_encoder.inverse_transform(model.predict(X))

    get_results(predictions, y, operations)


if __name__ == '__main__':
    main()
