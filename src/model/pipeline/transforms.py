"""This module provides data preprocessing utility."""

from math import ceil

import torch
from PIL.Image import Image
from torchvision.transforms.functional import pad, resize


def resize_pad(img: Image, smaller_side_size: int) -> torch.Tensor:
    """Resizes and pads an image.

    Resize an image to a given length based on the smallest side.
    Always keep the aspect ratio and make the image square by
    padding any excess pixels using black, equally distributed
    amongst both ends.

    Args:
        img (Image): The image as a PIL Image subclass.
        smaller_side_size (int): Size in pixels to be set as the smaller side.

    Returns:
        torch.Tensor: The resized and padded image.
    """
    # Resize
    aspect_ratio = img.height / img.width
    if aspect_ratio > 1:
        size = (smaller_side_size, round(smaller_side_size/aspect_ratio))
    else:
        size = (round(smaller_side_size*aspect_ratio), smaller_side_size)
    img = resize(img, size)
    # Padding
    pad_horizontal = (smaller_side_size-img.width) / 2
    pad_vertical = (smaller_side_size-img.height) / 2
    return pad(
        img,
        padding_mode='constant',
        padding=(  # (left, top, right, bottom)
            ceil(pad_horizontal),
            ceil(pad_vertical),
            int(pad_horizontal),
            int(pad_vertical),
        ),
    )
