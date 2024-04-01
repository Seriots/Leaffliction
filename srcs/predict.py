#!/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as preprocessing
import pickle
import os
import numpy as np

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper


def load_all_image(path: str) -> list:
    """Get the max folder size at depth from path"""
    all_img = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                img = preprocessing.load_img(entry.path,
                                             target_size=(256, 256))

                img_array = preprocessing.img_to_array(img)
                # print(img_array.shape)

                # img_array = tf.expand_dims(img_array, 0)
                # print(img_array.shape)
                all_img.append((entry.name, img_array, img))
            elif entry.is_dir():
                all_img += load_all_image(entry.path)
        print(f"{os.path.basename(path)} done")
        return all_img


def main():
    """Main"""
    args_handler = ArgsHandler(
        'This program take an image as arguments en display some \
modifications on it',
        [
            ArgsObject('images_path', 'The path of the targeted images'),
        ],
        [
            OptionObject('help', 'Show this help message',
                         name='h',
                         expected_type=bool,
                         default=False,
                         check_function=display_helper
                         ),
            OptionObject('model', 'The model to use',
                         name='m',
                         expected_type=str,
                         default='model.keras',
                         ),
            OptionObject('plot', 'Plot the image',
                         name='p',
                         expected_type=bool,
                         default=False,),
            OptionObject('show', 'Show the prediction value',
                         name='s',
                         expected_type=bool,
                         default=False,)
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

    img_path = user_input['args'][0]
    model_path = user_input['model']

    try:
        all_image = load_all_image(img_path)
    except Exception as e:
        print(e)
        return

    try:
        labels = pickle.load(open('labels.pkl', 'rb'))
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(e)
        return

    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

    right_guesses = 0

    if not user_input['show']:
        path, images_array, a = zip(*all_image)
        predictions = model.predict(np.array(images_array))
        right_guesses = sum((labels[tf.argmax(score)] in name)
                            for (name, score) in zip(path, predictions))
    else:
        for name, img, real_img in all_image:

            score = model(np.expand_dims(img, axis=0))[0]
            is_right_guess = labels[tf.argmax(score)] in name

            C = GREEN if is_right_guess else RED
            print(f"{C}{name:^24} == {labels[tf.argmax(score)]:^15}\
 -> {100 * tf.reduce_max(score):.2f}%{END}")

            right_guesses += is_right_guess

            if user_input['plot']:
                c = 'green' if is_right_guess else 'red'
                plt.text(0, -17, os.path.basename(name), color=c, fontsize=25)
                plt.imshow(real_img)
                try:
                    plt.show()
                except KeyboardInterrupt:
                    break

    valid_percent = 100 * right_guesses / len(all_image)
    print(BOLD + f"{valid_percent:.2f}%" + END +
          " of images where correctly identified")


if __name__ == "__main__":
    main()
