import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import image

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper

from utils.image_modifier import image_flip, image_rotate, image_skew
from utils.image_modifier import image_shear, image_crop, image_distortion
from utils.image_modifier import image_blur, image_brightness
from utils.image_modifier import image_contrast, image_projective


def check_transformation(args_handler, u_ipt):
    """Set the default features if the user didn't provide any."""
    if 'transformation' not in u_ipt or u_ipt['transformation'] is None:
        u_ipt['transformation'] = ['normal', 'flip', 'crop',
                                   'blur', 'brightness',
                                   'contrast', 'rotate']
    elif u_ipt['transformation'] == ['*']:
        u_ipt['transformation'] = ['normal', 'flip', 'rotate', 'skew',
                                   'shear', 'crop', 'distortion', 'blur',
                                   'brightness', 'contrast', 'projective']
    else:
        for elem in u_ipt['transformation']:
            if elem not in ['normal', 'flip', 'rotate', 'skew',
                            'shear', 'crop', 'distortion', 'blur',
                            'brightness', 'contrast', 'projective']:
                raise ValueError(f"Transformation {elem} is not supported")
    return u_ipt


def get_transformation(original_img: np.ndarray, transfo_required: list):
    """Get the transformation required"""
    transformation = {
        'normal': None,
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

    transformation_applied = {}

    for elem in transfo_required:
        if elem in transformation:
            function = transformation[elem]
            if function is not None:
                transformation_applied[elem] = function(original_img)
            else:
                transformation_applied[elem] = original_img
    return transformation_applied


def plot_transformation(dict_image: dict):
    """Display the transformation"""
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title('Image Augmentation')
    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    size = len(dict_image)

    for i, (key, value) in enumerate(dict_image.items()):
        fig.add_subplot((size // 4) + 1, 4, i+1)
        plt.imshow(value)
        key = key.capitalize()
        plt.title(key)


def save_all_transformation(dict_image: dict, path: str):
    """Save all the transformation"""
    for key, value in dict_image.items():
        if key == 'normal':
            continue
        root, ext = os.path.splitext(path)
        key = key.capitalize()
        try:
            image.imsave(f"{root}_{key}{ext}", value)
        except Exception as e:
            print(f"Error while saving {key} transformation: {e}")


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
                         ),
            OptionObject('transformation', 'The transformation to apply',
                         name='t',
                         expected_type=list,
                         default=None,
                         check_function=check_transformation
                         ),
            OptionObject('save', 'Save all transformation',
                         name='s',
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

    img = image.imread(path)
    transformation = get_transformation(img, user_input['transformation'])

    if user_input['save']:
        save_all_transformation(transformation, path)

    plot_transformation(transformation)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupt by user")


if __name__ == "__main__":
    main()
