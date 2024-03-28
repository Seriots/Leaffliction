#!/bin/env python3

import os
import matplotlib.image as mplimg

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper

from utils.data_transformation import imgt_mask_background
from utils.data_transformation import imgt_clear_background


def remove_background(path: str, destination: str) -> list[tuple]:
    """Get the max folder size at depth from path"""

    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                try:
                    img = mplimg.imread(entry.path)
                    background_mask = imgt_mask_background(img)
                    img_c = imgt_clear_background(img, background_mask)
                    mplimg.imsave(os.path.join(destination, entry.name), img_c)
                    print(f"Image {destination}/{entry.name} Done")
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            elif entry.is_dir():
                next_destination = os.path.join(destination, entry.name)
                if not os.path.exists(next_destination):
                    os.makedirs(next_destination)
                remove_background(entry.path, next_destination)
        return


def main():
    """Main"""
    args_handler = ArgsHandler(
        """This program take a path as arguments and balance \
            it to make all folders have the same amount of files""",
        [
            ArgsObject('folder_path', 'The path of the targeted folder')
        ],
        [
            OptionObject('help', 'Show this help message',
                         name='h',
                         expected_type=bool,
                         default=False,
                         check_function=display_helper
                         ),
            OptionObject('destination', 'The destination folder',
                         name='d',
                         expected_type=str,
                         default="background_removed_images"
                         ),
        ],
        """This program works recursively in folders\n"""
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
    destination = user_input['destination']

    if not os.path.exists(destination):
        os.makedirs(destination)
    elif not os.path.isdir(destination):
        print("Destination is not a folder")
        return
    elif os.listdir(destination):
        print("Destination folder is not empty")
        return

    remove_background(path, destination)


if __name__ == "__main__":
    main()
