#from plantcv import plantcv as pcv
#import plantcv as plantcv

## Set global debug behavior to None (default), "print" (to file), 
## or "plot" (Jupyter Notebooks or X11)
#pcv.params.debug = "plot"

##img = pcv.readimage("images/Apple/Apple_Black_rot/image (1).JPG")

## Charger l'image
#img, path, filename = pcv.readimage("images/Apple/Apple_Black_rot/image (125).JPG")

## Convertir l'image en niveaux de gris
#gray_img = pcv.rgb2gray(img)

### Seuiller l'image pour obtenir les régions d'intérêt (ROIs)
#image_gaussian_blur = pcv.threshold.binary(gray_img, 128, 'light')
#image_gaussian_blur = pcv.gaussian_blur(image_gaussian_blur, ksize=(7, 7), sigma_x=0, sigma_y=None)

### Utiliser l'image de la ROI comme masque sur l'image pour obtenir la ROI avec plantcv
#image_mask = pcv.apply_mask(img, image_gaussian_blur, 'white')

#image_roi = pcv.roi.auto_grid(mask=image_mask, nrows=2, ncols=2, img=img)

#mask = pcv.roi.filter(mask=image_mask, roi=image_roi, roi_type="partial")


### Définir la ROI comme la plus grande région détectée
##roi_contour, roi_hierarchy = pcv.roi_objects(image, 'largest', obj_hierarchy, id_objects)

### Dessiner un rectangle autour de la région d'intérêt (ROI) sur l'image originale
##roi_image = pcv.object_composition(image, [roi_contour], obj_hierarchy)

### Enregistrer l'image avec la ROI dessinée
##pcv.print_image(roi_image, "roi_image.jpg")

### Afficher l'image avec la ROI
#pcv.display_image(roi_image)

import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import rembg
import sys
import numpy as np


def is_roi_border(x, y, roi_start_x, roi_start_y, roi_h, roi_w, roi_line_w):
    """
    Return true if the pixel in position x, y is the border of the rectangle
    defined by the roi parameters.
    The contour is the line of the rectangle, with a width of roi_line_w.

    :param x: The x position of the pixel
    :param y: The y position of the pixel
    :param roi_start_x: The x position of the roi rectangle start
    :param roi_start_y: The y position of the roi rectangle start
    :param roi_h: The height of the roi rectangle
    :param roi_w: The width of the roi rectangle
    :param roi_line_w: The width of the roi rectangle line
    """

    return (
        (
            roi_start_x <= x <= roi_start_x + roi_w and
            roi_start_y <= y <= roi_start_y + roi_line_w
        )
        or
        (
            roi_start_x <= x <= roi_start_x + roi_w and
            roi_start_y + roi_h - roi_line_w <= y <= roi_start_y + roi_h
        )
        or
        (
            roi_start_x <= x <= roi_start_x + roi_line_w and
            roi_start_y <= y <= roi_start_y + roi_h
        )
        or
        (
            roi_start_x + roi_w - roi_line_w <= x <= roi_start_x + roi_w and
            roi_start_y <= y <= roi_start_y + roi_h
        )
    )


def create_roi_image(
    image,
    masked,
    filled
):

    """
    Create an image with the ROI rectangle and the mask
    """

    # Create a region of interest (ROI) rectangle
    roi_start_x = 0
    roi_start_y = 0
    roi_w = image.shape[0]
    roi_h = image.shape[1]
    roi_line_w = 5
    roi = pcv.roi.rectangle(
        img=masked,
        x=roi_start_x,
        y=roi_start_y,
        w=roi_w,
        h=roi_h
    )

    # Create a mask based on the ROI
    kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type='partial')

    roi_image = image.copy()
    roi_image[kept_mask != 0] = (0, 255, 0)
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if is_roi_border(x, y, roi_start_x, roi_start_y,
                             roi_h, roi_w, roi_line_w):
                roi_image[x, y] = (0, 0, 255)

    return roi_image, kept_mask


def main():
    #pcv.params.debug = "plot"
    img, path, filename = pcv.readimage(sys.argv[1], mode='native')

    #plt.imshow(img)
    img_without_bg = rembg.remove(img)

    gray_img = pcv.rgb2gray_lab(img_without_bg, channel='l')
    image_bw = pcv.threshold.binary(gray_img, 35, 'light')
    filled = pcv.fill(bin_img=image_bw, size=500)

    image_gaussian_blur = pcv.gaussian_blur(filled, ksize=(3, 3))
    masked = pcv.apply_mask(img, image_gaussian_blur, 'white')

    roi_image, kept_mask = create_roi_image(img, masked, filled)

    analyzed_image = pcv.analyze.size(img, kept_mask)

    pseudolandmarks_image = pcv.visualize.pseudolandmarks(img, kept_mask, analyzed_image)

    plt.imshow(pseudolandmarks_image)
    plt.show()


main()