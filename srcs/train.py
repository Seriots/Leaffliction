#!/bin/env python3

import os
import matplotlib.image as mplimg
import numpy as np
import datetime
import tensorflow as tf
import pickle
from tensorflow.keras import models, layers

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper


def save_test_images(x_test, y_test, labels):
    if os.path.isdir('test_images'):
        for image in os.listdir('test_images'):
            try:
                os.remove('test_images/' + image)
            except Exception as e:
                print(f"Error while deleting {image}: {e}")
    else:
        os.mkdir('test_images')

    for (image, label, i) in zip(x_test, y_test, range(len(y_test))):
        try:
            mplimg.imsave(f"test_images/{labels[label]}_{i}.JPG", image)
        except Exception as e:
            print(f"Error while saving {image}: {e}")


def list_to_dict(all_image: list) -> dict:
    """Transform a list of tuple into a dict"""
    result = {}
    for path, image in all_image:
        result[path] = image

    return result


def load_all_image(path: str, depth: int) -> list:
    """Get the max folder size at depth from path"""
    if depth == 0:
        all_img = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    entry_path = entry.path
                    img = mplimg.imread(entry_path)
                    all_img.append(img)
            print(f"{os.path.basename(path)} done")
            return [(os.path.basename(path), all_img)]
    else:
        all_path = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    all_path += load_all_image(entry.path, depth - 1)
        return all_path


def check_ratio(args_handler, user_input):
    if user_input['validation-ratio'] <= 0 or \
       user_input['validation-ratio'] >= 1:
        raise ValueError("Validation ratio must be between 0 and 1")
    if user_input['test-ratio'] < 0 or \
       user_input['test-ratio'] >= 1:
        raise ValueError("Test ratio must be between 0 and 1")
    if user_input['test-ratio'] + user_input['validation-ratio'] >= 1:
        raise ValueError("Sum of ratios can't be greater than 1")
    return user_input


def main():
    """Main"""
    args_handler = ArgsHandler(
        'This program take an image as arguments en display some \
modifications on it',
        [
            ArgsObject('image_path', 'The path of the targeted image')
        ],
        [
            OptionObject('help', 'Show this help message',
                         name='h',
                         expected_type=bool,
                         default=False,
                         check_function=display_helper
                         ),
            OptionObject('depth', 'The folder depth to count the data',
                         name='d',
                         expected_type=int,
                         default=1,
                         ),
            OptionObject('model', 'The model to save',
                         name='m',
                         expected_type=str,
                         default='model.keras',
                         ),
            OptionObject('validation-ratio', 'The ratio of the train data',
                         name='v',
                         expected_type=float,
                         default=0.8,
                         check_function=check_ratio
                         ),
            OptionObject('test-ratio', 'The ratio of validation data',
                         name='t',
                         expected_type=float,
                         default=0.0,
                         check_function=check_ratio
                         ),
            OptionObject('epochs', 'The number of epoch',
                         name='e',
                         expected_type=int,
                         default=10
                         ),
            OptionObject('start-weights', 'Initial weights',
                         name='w',
                         expected_type=str,
                         default=None)
        ],
        """"""
    )

    try:
        user_input = args_handler.parse_args()
        args_handler.check_args(user_input)
    except SystemExit:
        return
    except Exception as e:
        print(e)
        return

    path = user_input['args'][0]
    depth = user_input['depth']

    try:
        all_image = load_all_image(path, depth)
        all_image_dict = list_to_dict(all_image)
    except Exception as e:
        print(e)
        return

    model = models.Sequential()

    model.add(tf.keras.Input(shape=(256, 256, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(all_image_dict.keys()), activation='softmax'))
    model.summary()
    
    if 'start-weights' in user_input:
        try:
            weights = user_input['start-weights']
            if (weights):
                model.load_weights(weights)
        except Exception as e:
            print(f"Error while loading weights: {e}")
            return

    valid_ratio = user_input['validation-ratio']
    test_ratio = user_input['test-ratio']
    model_path = user_input['model']

    size = len(next(iter(all_image_dict.values())))
    trn_size = int(valid_ratio * size)
    test_size = int(test_ratio * size)

    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []

    x_rnd = [i for i in range(0, size)]
    np.random.shuffle(x_rnd)

    try:
        for i, value in enumerate(all_image_dict.values()):
            data = np.array(value)[x_rnd[:trn_size]]
            x_train.extend(data)
            y_train.extend([i] * (trn_size))
            test_data = np.array(value)[x_rnd[trn_size:trn_size + test_size]]
            x_test.extend(test_data)
            y_test.extend([i] * test_size)
            x_valid.extend(np.array(value)[x_rnd[trn_size + test_size:]])
            y_valid.extend([i] * (size - trn_size - test_size))
    except Exception as e:
        print(f"Error while splitting data: {e}")
        return

    labels = list(all_image_dict.keys())

    if x_test is not []:
        save_test_images(x_test, y_test, labels)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer='adam', loss=loss, metrics=['acc'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=3,
                                                  start_from_epoch=7)

    date = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    filepath = './weights/' + date + '/{epoch:02d}-{val_loss:.2f}.weights.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                    save_weights_only=True,
                                                    monitor='val_loss')

    valid_data = np.array(x_valid), np.array(y_valid)
    model.fit(np.array(x_train),
              np.array(y_train),
              validation_data=valid_data,
              epochs=user_input['epochs'],
              callbacks=[early_stop, checkpoint])
    try:
        model.save(model_path)
        pickle.dump(labels, open('labels.pkl', 'wb'))
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    main()
