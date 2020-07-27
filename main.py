"""
generate_dataset_to_file.py

Main interface to solve the ML Lunch Group problem number 5

Author: CJ Paterno
Date: 02/20/2020
"""

import keras

from data_generator import preprocess_input, generate_example_input
from constants import output_encoder


def main():

    some_input, some_label, operation = generate_example_input()

    X = preprocess_input(some_input)
    y = some_label

    model = keras.models.load_model('./my_model')

    predictions = output_encoder.inverse_transform(model.predict(X))

    sse = 0
    num_wrong = 0

    for i, result in enumerate(predictions):
        squared_error = (result[0] - y[i][0]) ** 2
        sse += squared_error

        if result != y[i][0]:
            print(f'Expected: {operation[i]} = {y[i][0]}\n'
                  f'Actual:   {result[0]}\n')
            num_wrong += 1

    percent_wrong = num_wrong / len(y) * 100

    print(f'Results\n'
          f'\tTotal Wrong   - {num_wrong} / {len(some_input)}\n'
          f'\tAccuracy      - {100 - percent_wrong:.2f}%\n'
          f'\tSSE           - {sse}')


if __name__ == '__main__':
    main()
