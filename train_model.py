from datetime import datetime

import keras
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential

from data_generator import generator, generate_images
from constants import output_encoder


def make_model(num_out=49):
    model = Sequential()

    model.add(Permute(
        (2, 3, 1),
        input_shape=(3, 28, 28)
    ))

    model.add(Conv2D(
        64,
        (16, 16),
        strides=(4, 4),
        activation='relu',
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
        units=num_out,
        activation='softmax'
    ))

    model.compile(
        optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'mse']
    )

    print(model.summary())

    return model


def train():
    epochs = 32
    steps_per_epoch = 1000
    batch_size = 128

    model = make_model()

    logdir = f"logs/scalars/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    val_x, val_y, _ = generate_images(steps_per_epoch)

    # Train my_model
    model.fit(
        generator(batch_size=batch_size),
        validation_data=(val_x, val_y),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback],
    )

    model.save('./my_model')


def test(model):
    print('\nTesting...\n')
    x, y, operation = generate_images(250000, output_encoder)
    predictions = model.predict(x)
    predictions = output_encoder.inverse_transform(predictions)
    y = output_encoder.inverse_transform(y)
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
          f'\tTotal Wrong   - {num_wrong} / {len(x)}\n'
          f'\tAccuracy      - {100 - percent_wrong:.2f}%\n'
          f'\tSSE           - {sse}')


if __name__ == '__main__':
    train()
    my_model = keras.models.load_model('./my_model')
    test(my_model)
