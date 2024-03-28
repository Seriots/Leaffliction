import rembg

from plantcv import plantcv as pcv


def imgt_gaussian_blur(mask, ksize=(3, 3)):
    """
    Apply a gaussian blur to an image

    :param image: The image to apply the blur to, The basic image
    :type image: np.ndarray
    :param ksize: The kernel size of the blur
    :type ksize: Tuple[int, int]
    :return: The image with the blur applied
    :rtype: np.ndarray
    """

    return pcv.gaussian_blur(mask, ksize)


def imgt_leaf_mask(img, mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    return pcv.apply_mask(img, mask, "white")


def imgt_mask_disease(img, background_mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """
    mask = pcv.threshold.dual_channels(img,
                                       x_channel="a",
                                       y_channel="b",
                                       points=[(55, 55), (100, 115)],
                                       above=True
                                       )

    img_mask = pcv.logical_xor(background_mask, mask)

    return img_mask


def imgt_mask_background(img):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    shadow_mask = pcv.rgb2gray_lab(img, channel='l')
    shadow_mask = pcv.threshold.binary(shadow_mask, 15, 'light')
    shadow_mask = pcv.fill(bin_img=shadow_mask, size=500)
    shadow_mask = pcv.erode(shadow_mask, 5, 1)

    img_withoutbg = rembg.remove(img)
    grey_scale = pcv.rgb2gray_lab(img_withoutbg, channel='l')
    mask_withoutbg = pcv.threshold.binary(grey_scale, 20, 'light')
    mask_withoutbg = pcv.logical_and(shadow_mask, mask_withoutbg)
    mask_withoutbg = pcv.fill_holes(bin_img=mask_withoutbg)

    return mask_withoutbg


def imgt_clear_background(img, mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    return pcv.apply_mask(img, mask, "white")


def imgt_roi(img, mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    roi = pcv.roi.rectangle(
        img=mask,
        x=0,
        y=0,
        w=mask.shape[0],
        h=mask.shape[1]
    )
    roi_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    border_size = 5

    img_roi = img.copy()
    img_roi[roi_mask != 0] = (0, 255, 0)
    img_roi[:border_size, :] = (0, 0, 255)
    img_roi[-border_size:, :] = (0, 0, 255)
    img_roi[:, :border_size] = (0, 0, 255)
    img_roi[:, -border_size:] = (0, 0, 255)

    return img_roi


def imgt_analyse(img, mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    roi = pcv.roi.rectangle(
        img=mask,
        x=0,
        y=0,
        w=mask.shape[0],
        h=mask.shape[1]
    )
    roi_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    analysis_img = pcv.analyze.size(img, roi_mask)

    return analysis_img


def imgt_x_pseudolandmarks(img, mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    roi = pcv.roi.rectangle(
        img=mask,
        x=0,
        y=0,
        w=mask.shape[0],
        h=mask.shape[1]
    )
    roi_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img, roi_mask)

    return top, bottom, center_v


def imgt_y_pseudolandmarks(img, mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    roi = pcv.roi.rectangle(
        img=mask,
        x=0,
        y=0,
        w=mask.shape[0],
        h=mask.shape[1]
    )
    roi_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(img, roi_mask)

    return left, right, center_h


def center_from_0_to_127(x, y):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    return [elem + 127 for elem in x], y


def from_0_100_to_0_255(x, y):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    return [elem * 255 / 100 for elem in x], y


def from_360_to_255(x, y):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """
    return x[:127], y[:127]


def imgt_color_histogram(img, mask):
    """
    return pixels of the image that are not in the mask

    :param img: The image to apply the mask to
    :type img: np.ndarray
    """

    all_frequencies = {
       'blue_frequencies': [None, 'blue'],
       'blue-yellow_frequencies': [center_from_0_to_127,
                                   (230 / 255, 230 / 255, 0)],
       'green_frequencies': [None, 'green'],
       'green-magenta_frequencies': [center_from_0_to_127,
                                     (230 / 255, 0, 230 / 255)],
       'hue_frequencies': [from_360_to_255, 'purple'],
       'lightness_frequencies': [from_0_100_to_0_255, 'grey'],
       'red_frequencies': [None, 'red'],
       'saturation_frequencies': [from_0_100_to_0_255,
                                  (0, 230 / 255, 230 / 255)],
       'value_frequencies': [from_0_100_to_0_255, 'orange'],
    }

    roi = pcv.roi.rectangle(
        img=mask,
        x=0,
        y=0,
        w=mask.shape[0],
        h=mask.shape[1]
    )
    roi_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    pcv.analyze.color(img, roi_mask, colorspaces='all', label='default')
    out = {}
    for key, value in all_frequencies.items():
        x = pcv.outputs.observations['default_1'][key]['label']
        y = pcv.outputs.observations['default_1'][key]['value']
        if value[0] is not None:
            x, y = value[0](x, y)
        out[key] = (x, y, value[1])

    return out, all_frequencies.keys()
