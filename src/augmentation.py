from PIL import Image
import numpy as np


def augment_image(
    image, size,
    zoom=None,
    rotation=None,
    flip_horizontal=None,
    flip_vertical=None,
):
    """
    Augment an image by applying a series of transformations.
    """

    image = augment_zoom(image, zoom)
    image = augment_rotate(image, rotation)
    image = augment_flip(image, flip_horizontal, flip_vertical)

    return image.resize(size, Image.BICUBIC)


def augment_zoom(image, zoom=1):
    """
    Zoom into an image by a given factor.
    """

    image = image.crop((
        (1 - zoom) * image.size[0] / 2,
        (1 - zoom) * image.size[1] / 2,
        (1 + zoom) * image.size[0] / 2,
        (1 + zoom) * image.size[1] / 2
    ))

    return image


def augment_rotate(image, rotation=0):
    """
    Rotate an image by a given angle.
    """

    image = image.rotate(rotation, resample=Image.BICUBIC,
                         fillcolor=(0, 0, 0), expand=True)

    return image


def augment_flip(image, flip_horizontal=False, flip_vertical=False):
    """
    Flip an image horizontally and/or vertically.
    """

    if flip_horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if flip_vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    return image


def augment_shift(image, shift_x=0, shift_y=0):
    """
    Shift an image by a given number of pixels.
    """

    image = image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, shift_x, 0, 1, shift_y)
    )

    return image


def augment_shear(image, shear):
    """
    Shear an image by a given angle.
    """

    image = image.transform(
        image.size,
        Image.AFFINE,
        (1, shear, 0, 0, 1, 0)
    )

    return image
