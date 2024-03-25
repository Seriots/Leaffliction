import numpy as np
import skimage
import augly.image as imaugs


def check_image(func):
    def inner(*args, **kwargs):
        if not isinstance(args[0], np.ndarray):
            raise TypeError("Input is not an image")
        return func(*args, **kwargs)
    return inner


@check_image
def image_flip(image: np.ndarray) -> np.ndarray:
    """Flip an image"""
    return imaugs.aug_np_wrapper(image, imaugs.hflip)


@check_image
def image_rotate(image: np.ndarray, angle: float = 25) -> np.ndarray:
    """Rotate an image"""
    return skimage.transform.rotate(image, angle)


@check_image
def image_skew(image: np.ndarray, skew: float = 2) -> np.ndarray:
    """Skew an image"""
    return imaugs.aug_np_wrapper(image, imaugs.skew, skew_factor=skew)


@check_image
def image_shear(image: np.ndarray, shear: float = 0.5) -> np.ndarray:
    """Shear an image"""
    t_form = skimage.transform.AffineTransform(shear=shear)
    return skimage.transform.warp(image, t_form.inverse)


@check_image
def image_crop(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
    """Crop an image"""
    if factor > 0.4:
        factor = 0.4
    elif factor < 0:
        factor = 0

    return imaugs.aug_np_wrapper(image, imaugs.crop,
                                 x1=factor,
                                 x2=1-factor,
                                 y1=factor,
                                 y2=1-factor
                                 )


@check_image
def image_distortion(image: np.ndarray) -> np.ndarray:
    """Distort an image"""
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 20
    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = skimage.transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] - 1.5 * 50
    out_cols = cols
    out = skimage.transform.warp(image,
                                 tform,
                                 output_shape=(out_rows, out_cols))
    return out


@check_image
def image_blur(image: np.ndarray, factor: float = 2) -> np.ndarray:
    """Blur an image"""
    return imaugs.aug_np_wrapper(image, imaugs.blur, radius=factor)


@check_image
def image_brightness(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Change brightness of an image"""
    return imaugs.aug_np_wrapper(image, imaugs.brightness, factor=factor)


@check_image
def image_contrast(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Change contrast of an image"""
    return imaugs.aug_np_wrapper(image, imaugs.contrast, factor=factor)


@check_image
def image_projective(image: np.ndarray) -> np.ndarray:
    """Project an image"""
    matrix = np.array([[1, -0.5, 100],
                       [0.1, 0.9, 50],
                       [0.0015, 0.0015, 1]])
    tform = skimage.transform.ProjectiveTransform(matrix=matrix)
    return skimage.transform.warp(image, tform.inverse)
