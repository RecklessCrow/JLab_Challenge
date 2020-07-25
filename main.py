"""
generate_dataset.py

Main interface to solve the ML Lunch Group problem number 5

Author: CJ Paterno
Date: 02/20/2020
"""

from datetime import datetime
from random import randint, choice

import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, normalize
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from mlxtend.data import loadlocal_mnist


def make_model(outputs) -> 'A Keras Model':
    model = Sequential()

    model.add(Permute(
        (2, 3, 1),
        input_shape=(3, 28, 28)
    ))

    model.add(Conv2D(
        64,
        (16, 16),
        strides=(4, 4),
        activation='relu'
    ))

    model.add(Conv2D(
        64,
        (4, 4),
        strides=(2, 2),
        activation='relu'
    ))

    model.add(Flatten())

    model.add(Dense(
        units=128,
        activation='relu'
    ))

    model.add(Dense(
        units=outputs,
        activation='softmax'
    ))

    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'mse']
    )

    return model


def generate_dataset(elements=500000):
    images, labels = loadlocal_mnist(
        images_path='train-images.idx3-ubyte',
        labels_path='train-labels.idx1-ubyte'
    )

    images = normalize(images)

    X = []
    y = []
    image_labels = []
    indexes = len(images) - 1
    op_encoder = LabelEncoder()
    operators = np.array(["+", "-", "*"])
    op_encoder.fit(operators)

    for i in range(elements):
        index_a = randint(0, indexes)
        index_b = randint(0, indexes)

        a_image = images[index_a]
        a_image = np.reshape(a_image, (-1, 28))
        a_label = labels[index_a]

        b_image = images[index_b]
        b_image = np.reshape(b_image, (-1, 28))
        b_label = labels[index_b]

        operator = choice(operators)

        image_labels.append((a_label, operator, b_label))

        op_array = np.zeros((28, 28))
        temp = op_encoder.transform([operator])[0]
        op_array.fill(temp)

        input_list = [a_image, op_array, b_image]

        X.append(input_list)

        label = eval(f'{a_label} {operator} {b_label}')
        y.append([label])

    X = np.array(X)
    y = np.array(y)

    return X, y, image_labels


def main():
    train_set_size = 500000
    test_set_size = 250

    logdir = "logs/scalars/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    data, labels, _ = generate_dataset(train_set_size)
    encoder = OneHotEncoder()
    labels = encoder.fit_transform(labels).toarray()

    model = make_model(len(labels[0]))

    # Train model
    model.fit(
        x=data,
        y=labels,
        epochs=50,
        validation_split=0.2,
        use_multiprocessing=True,
        callbacks=[tensorboard_callback],
    )

    # Test model
    data, labels, image_labels = generate_dataset(test_set_size)

    predictions = model.predict(data)
    predictions = encoder.inverse_transform(predictions)

    sse = 0
    num_wrong = 0
    for i in range(len(predictions)):
        squared_error = pow((predictions[i][0] - labels[i][0]), 2)
        sse += squared_error

        if predictions[i] != labels[i][0]:
            print(f'{i}) {image_labels[i][0]} {image_labels[i][1]} {image_labels[i][2]} = '
                  f'{predictions[i][0]} -> Actual: {labels[i][0]}')
            num_wrong += 1

    print(f'Total wrong: {num_wrong}/{test_set_size}\n'
          f'Sum Squared Error: {sse}')

    # todo: generate CSV for inputs, then encode it for the model


if __name__ == '__main__':
    main()
