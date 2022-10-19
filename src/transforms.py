import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    ToTensord,
    Resized,
    RandHistogramShiftd,
    RandGaussianNoised,
)
from monai.data.image_reader import ITKReader

from normalize import Normalized
from clahe import Clahed


def label_to_np(input, key="label"):
    input[key] = np.array([input[key]])
    return input


def move_channels_first(input, key="image"):
    # move the channel axis to the first position
    input[key] = np.transpose(input[key], (2, 1, 0))
    return input


def grey_to_rgb_sv(input, key="image"):
    # repeat the single channel image 3 times to make it RGB
    input[key] = input[key].repeat(3, 1, 1)
    # shape is now (3, 256, 256)
    return input


def grey_to_rgb_mv(input, key="image"):
    # add a channel axis to the image (before the 7 views were stacked in the channel axis)
    # repeat the single channel image 3 times to make it RGB
    input[key] = input[key].unsqueeze(1).repeat(1, 3, 1, 1)
    # shape is now (7, 3, 256, 256)
    return input


def get_transforms(multi_view=False, augmentations=True):

    """
    Build the data pipeline from path and integer label to tensor input for the model
    """

    transforms = []

    transforms.append(label_to_np)

    transforms.append(LoadImaged(keys=["image"], reader=ITKReader, image_only=True))

    transforms.append(move_channels_first)

    transforms.append(Resized(keys=["image"], spatial_size=[227, 227]))

    transforms.append(
        Clahed(
            keys=["image"],
            kernel_size=None,
            contrast_clip_limit=0.015,
            nbins=256,
            dtype=np.float32,
        )
    )

    transforms.append(
        Normalized(
            keys=["image"],
            a_upper_quant=0.95,
            a_lower_quant=0.05,
            b_upper_value=1.0,
            b_lower_value=0.0,
            dtype=np.float32,
        )
    )

    if augmentations:
        transforms.append(
            RandAffined(
                keys=["image"],
                prob=0.8,
                rotate_range=[8, 8],
                shear_range=[0.025, 0.025],
                translate_range=[20, 20],
                scale_range=[0.05, 0.05],
            )
        )
        transforms.append(RandHistogramShiftd(keys=["image"], prob=0.5, num_control_points=20))
        transforms.append(RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.05))

    transforms.append(ToTensord(keys=["image", "label"], dtype=torch.float))

    transforms.append(grey_to_rgb_mv if multi_view else grey_to_rgb_sv)

    return Compose(transforms)
