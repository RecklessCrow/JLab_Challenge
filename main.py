"""
generate_dataset_to_file.py

Main interface to solve the ML Lunch Group problem number 5

Author: CJ Paterno
Date: 02/20/2020
"""

import keras

from constants import output_encoder
from data_generator import preprocess_input, generate_example_input
from train_model import get_results


def main():
    """
    Main function
    :return:
    """

    X, y, operations = generate_example_input()

    X = preprocess_input(X)

    model = keras.models.load_model('./models/2020-07-27_14:49')

    predictions = output_encoder.inverse_transform(model.predict(X))

    get_results(predictions, y, operations)


if __name__ == '__main__':
    main()
