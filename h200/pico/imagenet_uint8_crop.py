# ruff: noqa: ANN001, ANN201, ANN202, ANN204
"""PIL-compatible bicubic RandomResizedCrop for decoded CPU uint8 tensors."""

import ctypes
from pathlib import Path

import torch
from torchvision.transforms import RandomResizedCrop


def _load_crop():
    library = ctypes.CDLL(str(Path(__file__).with_name("imagenet_uint8_bicubic.so")))
    crop = library.va_crop_bicubic
    crop.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_int64] * 3 + [ctypes.c_int] * 3 + [ctypes.c_bool]
    )
    crop.restype = None
    return crop


class Uint8BicubicCropAndFlip:
    def __init__(self, crop):
        self.scale, self.ratio = crop.scale, crop.ratio

    def __call__(self, image):
        top, left, height, width = RandomResizedCrop.get_params(image, self.scale, self.ratio)
        flipped = bool(torch.rand(1) < 0.5)
        crop = image[:, top : top + height, left : left + width]
        return (bicubic_224(crop, flipped), False)


_crop = _load_crop()


def bicubic_224(image, flipped):
    result = torch.empty((3, 224, 224), dtype=torch.uint8)
    _crop(image.data_ptr(), result.data_ptr(), *image.stride(), *image.shape[-2:], 224, flipped)
    return result
