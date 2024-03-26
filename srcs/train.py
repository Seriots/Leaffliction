#!/bin/env python3

import os
import matplotlib.image as mplimg
import tensorflow as tf
from tensorflow.keras import models, layers

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper

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
    print(len(all_image_dict))

    model = models.Sequential()
    size = all_image_dict['Apple_scab'][0].shape[0]
    print(size)
    model.add(layers.Conv2D(size, (3, 3), activation='relu', input_shape=(size, size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(all_image_dict)))
    model.summary()

    nb_data_per_class = 100
    x_train, y_train = [], []
    for key, value in all_image_dict.items():
        x_train += value[:nb_data_per_class]
        y_train += [key] * nb_data_per_class

    model
    # model.fit()

    

if __name__ == "__main__":
    main()
