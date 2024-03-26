#!/bin/env python3

import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
# import plantcv
from plantcv import plantcv as pcv
from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper

def image_gaussian_blur(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    black_white = np.zeros_like(gray)
    black_white[gray < 128] = 0
    black_white[gray >= 128] = 255

    return pcv.gaussian_blur(black_white, ksize=(11, 11), sigma_x=0, sigma_y=None)

def main():
    """Main"""

    args_handler = ArgsHandler(
    'This program take an image as arguments en display some \
tranformation on it',
    [
        ArgsObject('data_path', 'The path of the targeted image or targeted folder')
    ],
    [
        OptionObject('help', 'Show this help message',
                        name='h',
                        expected_type=bool,
                        default=False,
                        check_function=display_helper
                        ),
        OptionObject('source', 'The path of the source folder',
                        name='s',
                        expected_type=str,
                        default=None
                        ),
        OptionObject('destination', 'The path of the destination folder',
                        name='d',
                        expected_type=str,
                        default=None
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
    
    if len(user_input['args']) == 1:
        path = user_input['args'][0]
    elif 'source' in user_input and user_input['source'] and 'destination' in user_input and user_input['destination']:
        path = user_input['source']
    else:
        print("No valid path given. If you provide a source please give a destination too.")
        return

    img = mplimg.imread(path)
    fig = plt.figure(figsize=(6, 4))
    fig.canvas.manager.set_window_title('Image Transformation')
    fig.subplots_adjust(wspace=0.3, hspace=0.1)
    fig.add_subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('Original')

    img_gaussianblur = image_gaussian_blur(img)

    fig.add_subplot(2, 3, 2)
    plt.imshow(img_gaussianblur, cmap='gray')
    plt.title('Gaussian Blur')


    pcv.params.sample_label = "plant"
    gray_img = pcv.rgb2gray(img)
    thresh1 = pcv.threshold.binary(gray_img, 35)
    thresh2 = pcv.threshold.dual_channels(img, x_channel = "a", y_channel = "b", points = [(80,80),(125,140)], above=True)
    fill_img = pcv.fill(bin_img=thresh2, size=60)
    # fill_img = pcv.fill_holes(fill_img)
    roi = pcv.roi.rectangle(img, 0, 0, img.shape[0], img.shape[1])
    mask = pcv.roi.filter(fill_img, roi, roi_type='partial')
    analysis_img = pcv.analyze.size(img, mask)
    color_histo = pcv.analyze.color(img, mask)

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)
    bottom_landmarks = pcv.outputs.observations['plant']['bottom_lmk']['value']
    # roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi, roi_hierarchy, objects, obj_hierarchy, device, debug="print")
    # print(bottom_landmarks)
    # print(bottom)
    fig.add_subplot(2, 3, 3)
    plt.imshow(analysis_img, cmap='gray')
    plt.title('analyse')

    img_copy = img.copy()
    # cv2.drawContours(img_copy, top, -1, (255, 0, 0), pcv.params.line_thickness)
    # cv2.drawContours(img_copy, bottom_landmarks, -1, (0, 0, 255), pcv.params.line_thickness)
    # cv2.drawContours(img_copy, center_v, -1, (0, 255, 0), pcv.params.line_thickness)


    fig.add_subplot(2, 3, 4)
    plt.imshow(mask, cmap='gray')
    # plt.plot(chain)
    plt.title('Mask')

    fig.add_subplot(2, 3, 5)
    # print(	pcv.outputs.observations['plant_1'])
    # x, y = zip(bottom)
    # print(x)
    x = [d[0][0] for d in bottom] + [d[0][0] for d in top] + [d[0][0] for d in center_v] + [d[0][0] for d in left] + [d[0][0] for d in right] + [d[0][0] for d in center_h]
    y = [d[0][1] for d in bottom] + [d[0][1] for d in top] + [d[0][1] for d in center_v] + [d[0][1] for d in left] + [d[0][1] for d in right] + [d[0][1] for d in center_h]
    plt.imshow(img_copy)
    plt.scatter(x=x, y=y)
    plt.title('acute')


    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
