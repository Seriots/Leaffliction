import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as preprocessing
import pickle
import os

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

                img_array = tf.expand_dims(img_array, 0)

                all_img.append((entry.path, img_array, img))
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

    for name, img, real_img in all_image:
        predictions = model.predict(img)
        score = tf.nn.softmax(predictions[0])

        print(f"{name} == {labels[tf.argmax(score).numpy()]}\
-> {100 * tf.reduce_max(score):.2f}%")

        if user_input['plot']:
            color = 'green' if labels[tf.argmax(score).numpy()]\
                in name else 'red'
            plt.text(0, -17, os.path.basename(name), color=color, fontsize=25)
            plt.imshow(real_img)
            try:
                plt.show()
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
