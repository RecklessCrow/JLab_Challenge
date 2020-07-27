from datetime import datetime
from os.path import isdir

import keras
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential

from constants import output_encoder
from data_generator import generator, generate_testing_images


def make_model(num_out=49):
    model = Sequential()

    model.add(Permute(
        (2, 3, 1),
        input_shape=(3, 28, 28)
    ))

    model.add(Conv2D(
        30,
        (20, 20),
        strides=(2, 2),
        activation='relu',
    ))

    model.add(Conv2D(
        90,
        (5, 5),
        strides=(2, 2),
        activation='relu'
    ))

    model.add(Flatten())

    model.add(Dense(
        units=300,
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


curr_date = datetime.now().strftime("%Y-%m-%d_%H:%M")


def train(model_file=None):
    epochs = 100
    steps_per_epoch = 100
    batch_size = epochs * steps_per_epoch

    # Create model
    # Load checkpoint if exists
    previous_checkpoint = f'./checkpoints/2020-07-27_14:30'
    if isdir(previous_checkpoint):
        model = keras.models.load_model(previous_checkpoint)

    else:
        model = make_model()

    print(model.summary())

    # Create callbacks
    log_dir = f'./logs/scalars/{curr_date}'
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_file = f'./checkpoints/{curr_date}'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_file)

    # Generate validation set
    val_x, val_y, _ = generate_testing_images(gen_mode=1)

    # Train model
    model.fit(
        generator(batch_size=batch_size),
        validation_data=(val_x, val_y),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback, checkpoint_callback],
    )

    score = test(model)

    if model_file is None:
        model.save(f'./models/{score}')
    else:
        model.save(model_file)


def test(model):

    print('Testing...')

    x, y, operations = generate_testing_images()
    predictions = model.predict(x)
    predictions = output_encoder.inverse_transform(predictions)
    y = output_encoder.inverse_transform(y)

    sse = get_results(predictions, y, operations)

    return sse


def get_results(actual, expected, operations):
    sse = 0
    num_wrong = 0

    for i, result in enumerate(actual):
        squared_error = (result[0] - expected[i][0]) ** 2
        sse += squared_error

        if result != expected[i][0]:
            print(f'Expected: {operations[i]} = {expected[i][0]}\n'
                  f'Actual:   {result[0]}\n')
            num_wrong += 1

    percent_wrong = num_wrong / len(expected) * 100

    print(f'Results\n'
          f'\tTotal Wrong   - {num_wrong} / {len(x)}\n'
          f'\tAccuracy      - {100 - percent_wrong:.2f}%\n'
          f'\tSSE           - {sse}')

    return sse


if __name__ == '__main__':

    train()
