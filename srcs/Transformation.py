#!/bin/env python3

import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import os

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper

from utils.data_transformation import imgt_mask_disease, imgt_mask_background
from utils.data_transformation import imgt_gaussian_blur, imgt_leaf_mask
from utils.data_transformation import imgt_roi, imgt_y_pseudolandmarks
from utils.data_transformation import imgt_x_pseudolandmarks, imgt_analyse
from utils.data_transformation import imgt_color_histogram


def check_transformation(args_handler, inpt_u):
    """Modify the transformation input of the user"""
    if 'transformation' not in inpt_u or inpt_u['transformation'] is None:
        inpt_u['transformation'] = ['background', 'gaussian-blur', 'mask',
                                    'roi', 'analyse',
                                    'pseudolandmarks-x', 'pseudolandmarks-y']
    else:
        for elem in inpt_u['transformation']:
            if elem not in ['background', 'gaussian-blur', 'mask',
                            'roi', 'analyse',
                            'pseudolandmarks-x', 'pseudolandmarks-y']:
                raise ValueError(f"Transformation {elem} is not supported")
    return inpt_u


def manage_path_input(user_input):

    destination = None
    if len(user_input['args']) == 1:
        path = user_input['args'][0]
        if (os.path.isfile(path)):
            path_type = 'file'
        else:
            raise ValueError(f"Path {path} is not a file")
    elif 'source' in user_input \
            and user_input['source'] \
            and 'destination' in user_input \
            and user_input['destination']:
        path = user_input['source']
        destination = user_input['destination']
        if (os.path.isdir(path)):
            path_type = 'folder'
        else:
            raise ValueError(f"Path {path} is not a folder")
        if os.path.exists(destination):
            if not os.path.isdir(destination):
                raise ValueError(f"Destination {destination} is not a folder")
        else:
            os.makedirs(destination)
    else:
        raise ValueError("No valid path given.\
If you provide a source please give a destination too.")
    return path, path_type, destination


def apply_transformation(path,
                         transformation_list,
                         color_histogram,
                         dest=None,
                         save=False
                         ):
    """Apply all transformation to the image"""
    try:
        img = mplimg.imread(path)
    except Exception as e:
        print(e)
        return

    fig = plt.figure(figsize=(8, 4))
    fig.canvas.manager.set_window_title('Image Transformation')
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    subplots_size = len(transformation_list) // 4 + 1
    fig.add_subplot(subplots_size, 4, 1)

    plt.imshow(img)
    plt.title('Original')

    background_mask = imgt_mask_background(img)
    disease_mask = imgt_mask_disease(img, background_mask)

    for i, transformation in enumerate(transformation_list):
        fig.add_subplot(subplots_size, 4, i + 2)
        if transformation == 'background':
            plt.imshow(background_mask, cmap='gray')
            plt.title('Background mask')
        elif transformation == 'gaussian-blur':
            gaussian_img = imgt_gaussian_blur(disease_mask, ksize=(7, 7))
            plt.imshow(gaussian_img, cmap='gray')
            plt.title('Gaussian Blur')
        elif transformation == 'mask':
            mask_img = imgt_leaf_mask(img, disease_mask)
            plt.imshow(mask_img, cmap='gray')
            plt.title('disease mask')
        elif transformation == 'roi':
            roi_img = imgt_roi(img, disease_mask)
            plt.imshow(roi_img)
            plt.title('ROI')
        elif transformation == 'analyse':
            analyse_img = imgt_analyse(img, disease_mask)
            plt.imshow(analyse_img)
            plt.title('Analyse')
        elif transformation == 'pseudolandmarks-x':
            top, bot, ctrv = imgt_x_pseudolandmarks(img, disease_mask)
            plt.imshow(img)
            plt.scatter(x=[d[0][0] for d in bot], y=[d[0][1] for d in bot],
                        color=(253 / 255, 1 / 255, 255 / 255))
            plt.scatter(x=[d[0][0] for d in top], y=[d[0][1] for d in top],
                        color=(2 / 255, 34 / 255, 255 / 255))
            plt.scatter(x=[d[0][0] for d in ctrv], y=[d[0][1] for d in ctrv],
                        color=(255 / 255, 79 / 255, 0 / 255))
            plt.title('Pseudolandmarks X')
        elif transformation == 'pseudolandmarks-y':
            lft, rht, ctrh = imgt_y_pseudolandmarks(img, disease_mask)
            plt.imshow(img)
            plt.scatter(x=[d[0][0] for d in lft], y=[d[0][1] for d in lft],
                        color=(253 / 255, 1 / 255, 255 / 255))
            plt.scatter(x=[d[0][0] for d in rht], y=[d[0][1] for d in rht],
                        color=(2 / 255, 34 / 255, 255 / 255))
            plt.scatter(x=[d[0][0] for d in ctrh], y=[d[0][1] for d in ctrh],
                        color=(255 / 255, 79 / 255, 0 / 255))
            plt.title('Pseudolandmarks Y')

    if save:
        try:
            fig.savefig(f"{dest}/{os.path.basename(path)[:-4]}_transfo.png")
        except Exception as e:
            print(e)
        plt.close(fig)

    if color_histogram:
        fig2 = plt.figure(figsize=(8, 4))
        fig2.canvas.manager.set_window_title('Image Color Histogram')

        c_hist, all_freq = imgt_color_histogram(img, disease_mask)

        for color in c_hist:
            plt.plot(c_hist[color][0],
                     c_hist[color][1],
                     color=c_hist[color][2]
                     )
        plt.xlabel('Pixel Intensity')
        plt.xticks(range(0, 256, 25))
        plt.ylabel('Proportions of pixles (%)')
        plt.legend(all_freq, loc='upper left')

        if save:
            try:
                fig2.savefig(f"{dest}/{os.path.basename(path)[:-4]}_hist.png")
            except Exception as e:
                print(e)
            plt.close(fig2)


def apply_transformation_folder(path, transfo, c_hist, dest):
    """Apply all transformation to the image in the folder"""
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                apply_transformation(entry.path, transfo, c_hist, dest, True)
            elif entry.is_dir():
                apply_transformation_folder(entry.path, transfo, c_hist, dest)


def main():
    """Main"""

    args_handler = ArgsHandler(
                    'This program take an image as \
arguments and display some tranformation on it',
                    [
                        ArgsObject('data_path',
                                   'The path of the targeted image \
or targeted folder',
                                   Optional=True)
                    ],
                    [
                        OptionObject('help', 'Show this help message',
                                     name='h',
                                     expected_type=bool,
                                     default=False,
                                     check_function=display_helper
                                     ),
                        OptionObject('source', 'The path of the source folder',
                                     name='src',
                                     expected_type=str,
                                     default=None
                                     ),
                        OptionObject('destination', 'The path of the \
destination folder',
                                     name='dest',
                                     expected_type=str,
                                     default=None
                                     ),
                        OptionObject('transformation',
                                     """The transformation to apply to \
the image, available transormation are:
                                                background, gaussian-blur,
                                                mask, roi, analyse,
                                                pseudolandmarks-x,
                                                pseudolandmarks-y""",
                                     name='t',
                                     expected_type=list,
                                     default=None,
                                     check_function=check_transformation
                                     ),
                        OptionObject('color-histogram', 'Display the color \
histogram of the image',
                                     name='c',
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

    try:
        path, path_type, destination = manage_path_input(user_input)
    except Exception as e:
        print(e)
        return

    transformation_list = user_input['transformation']
    color_histogram = user_input['color-histogram']

    if path_type == 'file':
        apply_transformation(path,
                             transformation_list,
                             color_histogram)
    else:
        apply_transformation_folder(path,
                                    transformation_list,
                                    color_histogram,
                                    destination)

    try:
        if path_type == 'file':
            plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
