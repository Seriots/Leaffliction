#!/bin/env python3

import os
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def list_to_dict(all_image: list) -> dict:
    """Transform a list of tuple into a dict"""
    result = {}
    for path, image in all_image:
        result[path] = image

    return result


def load_all_image(path: str, depth: int) -> list[tuple]:
    """Get the max folder size at depth from path"""

    if depth == 0:
        all_img = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    entry_path = entry.path
                    img = mplimg.imread(entry_path)
                    # img = rgb2gray(img)
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
                         default=2,
                         ),
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
        print("to dict done")
    except Exception as e:
        print(e)
        return

    model = models.Sequential()

    size = all_image_dict['Apple_scab'][0].shape[0]
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
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
    model.add(layers.Dense(4))
    model.summary()

    valid_ratio = 0.9
    size = len(next(iter(all_image_dict.values())))
    train_size = int(valid_ratio * size)
    x_train, x_valid = [], []
    y_train, y_valid = [], []
    x_rnd = [i for i in range(0, size)]
    np.random.shuffle(x_rnd)
    i = 0
    for key, value in all_image_dict.items():
        data = np.array(value)[x_rnd[:train_size]]
        x_train.extend(data)
        y_train.extend([i] * train_size)
        x_valid.extend(np.array(value)[x_rnd[train_size:]])
        y_valid.extend([i] * (size - train_size))
        i += 1
    labels = all_image_dict.keys()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, start_from_epoch=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./weights/{epoch:02d}-{val_loss:.2f}.weights.h5', save_weights_only=True, monitor='val_loss')
    valid_data = np.array(x_valid), np.array(y_valid)
    model.fit(np.array(x_train), np.array(y_train), validation_data = valid_data, epochs=20, callbacks = [early_stop, checkpoint])
    model.save('./model.keras')

    

if __name__ == "__main__":
    main()
