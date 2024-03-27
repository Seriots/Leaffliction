
from plantcv import plantcv as pcv
import rembg


def gaussian_blur(img, ksize=(3, 3)):
    """
    Apply a gaussian blur to an image

    :param image: The image to apply the blur to, The basic image
    :type image: np.ndarray
    :param ksize: The kernel size of the blur
    :type ksize: Tuple[int, int]
    :return: The image with the blur applied
    :rtype: np.ndarray
    """
    img_bw =  pcv.threshold.dual_channels(img, x_channel = "a", y_channel = "b", points = [(55,55),(100,115)], above=True)

    return pcv.gaussian_blur(img_bw, ksize)