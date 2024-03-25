import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np

import plantcv
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

    device, roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi, roi_hierarchy, objects, obj_hierarchy, device, debug="print")


    fig.add_subplot(2, 3, 3)
    plt.imshow(img_mask, cmap='gray')
    plt.title('Gaussian Blur')

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
