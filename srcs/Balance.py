#!/bin/env python3

import os
import random
import matplotlib.image as mplimg

from tqdm import tqdm
from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper

from utils.data_augmentation import image_flip, image_rotate, image_skew
from utils.data_augmentation import image_shear, image_crop, image_distortion
from utils.data_augmentation import image_blur, image_brightness
from utils.data_augmentation import image_contrast, image_projective


def check_transformation(args_handler, u_ipt):
    """Set the default features if the user didn't provide any."""
    if 'transformation' not in u_ipt or u_ipt['transformation'] is None:
        u_ipt['transformation'] = ['flip', 'blur', 'brightness',
                                   'contrast', 'rotate']
    elif u_ipt['transformation'] == ['*']:
        u_ipt['transformation'] = ['flip', 'rotate', 'skew',
                                   'shear', 'crop', 'distortion', 'blur',
                                   'brightness', 'contrast', 'projective']
    else:
        for elem in u_ipt['transformation']:
            if elem not in ['flip', 'rotate', 'skew',
                            'shear', 'crop', 'distortion', 'blur',
                            'brightness', 'contrast', 'projective']:
                raise ValueError(f"Transformation {elem} is not supported")
    return u_ipt


def check_mode(args_handler, u_ipt):
    """Set the default features if the user didn't provide any."""
    if 'mode' not in u_ipt or u_ipt['mode'] is None:
        u_ipt['mode'] = 'balance'
    elif u_ipt['mode'] != 'balance' and u_ipt['mode'] != 'clear':
        raise ValueError(f"Mode are 'balance' or 'clear' not {u_ipt['mode']}")
    return u_ipt


def list_to_dict(all_path_size: list) -> dict:
    """Transform a list of tuple into a dict"""
    result = {}
    for path, size in all_path_size:
        result[path] = size

    return result


def get_max_size(path: str, depth: int) -> list:
    """Get the max folder size at depth from path"""

    if depth == 0:
        count = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    count += 1
            return [(path, count)]
    else:
        all_path = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    all_path += get_max_size(entry.path, depth - 1)
        return all_path


def get_allowed_transfo(img_path: str, allowed_transfo: list) -> str:
    """Get a random transformation not currently existing in the folder"""
    valid = []
    for transfo in allowed_transfo:
        if not os.path.exists(f"{img_path[:-4]}_{transfo.capitalize()}.JPG"):
            valid.append(transfo)
    if len(valid) == 0:
        return None
    return valid[random.randint(0, len(valid) - 1)]


def balance_data(all_path: dict, max_size: int, allowed_transfo: list):
    """Balance image in all folder to reach the amount of max_size
    Image are balanceed with augmentation function"""
    augmentation = {
        'flip': image_flip,
        'rotate': image_rotate,
        'skew': image_skew,
        'shear': image_shear,
        'crop': image_crop,
        'distortion': image_distortion,
        'blur': image_blur,
        'brightness': image_brightness,
        'contrast': image_contrast,
        'projective': image_projective
    }
    for key, value in all_path.items():
        print(f"Balance folder {key}")
        list_file = os.listdir(key)
        for _ in tqdm(range(max_size - value)):
            aug_use = None
            while aug_use is None and len(list_file) > 0:
                rimg = random.choice(list_file)
                img_path = f"{key}/{rimg}"

                aug_use = get_allowed_transfo(img_path, allowed_transfo)
                if aug_use is None:
                    list_file.remove(rimg)
            if aug_use is None:
                break

            transfo = augmentation[aug_use]

            img = mplimg.imread(img_path)
            new_img = transfo(img)

            root, ext = os.path.splitext(img_path)
            try:
                mplimg.imsave(f"{root}_{aug_use.capitalize()}{ext}", new_img)
            except Exception as e:
                print(f"Error while saving {key} transformation: {e}")


def clear_data(all_path: dict, clear_transfo: list):
    """Clear the augmented data.
    Only Augmentation in allowed_transo are cleared"""

    for key, value in all_path.items():
        with os.scandir(key) as it:
            for entry in it:
                for transfo in clear_transfo:
                    if entry.name.endswith(f"{transfo.capitalize()}.JPG"):
                        os.remove(entry.path)


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
            OptionObject('depth', 'The folder depth to count the data',
                         name='d',
                         expected_type=int,
                         default=1,
                         ),
            OptionObject('transformation', 'The transformation to apply',
                         name='t',
                         expected_type=list,
                         default=None,
                         check_function=check_transformation
                         ),
            OptionObject('mode', 'Mode to use: balance or clear',
                         name='m',
                         expected_type=str,
                         default='balance',
                         check_function=check_mode
                         )
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
    depth = user_input['depth']
    allowed_transfo = user_input['transformation']
    mode = user_input['mode']

    try:
        all_path = list_to_dict(get_max_size(path, depth))
    except Exception as e:
        print(e)
        return

    if mode == 'balance':
        if len(all_path.items()) == 0:
            print("Error in depth alignement")
            return
        max_size = max(all_path.values())

        try:
            balance_data(all_path, max_size, allowed_transfo)
        except Exception as e:
            print(e)
            return
    else:
        clear_data(all_path, allowed_transfo)


if __name__ == "__main__":
    main()
