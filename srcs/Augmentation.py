import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper


def get_file_count(path: str) -> int:
    """Get a folder and recursively count all files in it"""
    count = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                count += get_file_count(entry.path)
            else:
                count += 1
    return count


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
            OptionObject('debug', 'Launch in debug mode',
                         name='d',
                         expected_type=bool,
                         default=False
                         )
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

    img = mpimg.imread(path)

    if not user_input['debug']:
        mpimg.imsave('test.jpg', img)
    else:
        plt.imshow(img)
        try:
            plt.show()
        except KeyboardInterrupt:
            print("Interrupt by user")


if __name__ == "__main__":
    main()
