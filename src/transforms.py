import torch
import numpy as np
from copy import deepcopy
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

from src.normalize import Normalized
from src.clahe import Clahed


def label_to_np(input, key="label"):
    # create numpy array from label int
    output = deepcopy(input)
    output[key] = np.array([output[key]])
    return output


def move_channels_first(input, key="image"):
    # move the channel axis to the first position
    # because contrary to the description in the ITKReader, it outputs as WHC
    input[key] = np.transpose(input[key], (2, 1, 0))
    return input


def grey_to_rgb_sv(input, key="image"):
    # repeat the single channel image 3 times to make it RGB
    # (1, 227, 227) -> (3, 227, 227)
    input[key] = input[key].repeat(3, 1, 1)
    return input


def grey_to_rgb_mv(input, key="image"):
    # add a channel axis to the image (before the 7 views were stacked in the channel axis)
    # repeat the single channel image 3 times to make it RGB
    # (7, 227, 227) -> (7, 3, 227, 227)
    input[key] = input[key].unsqueeze(1).repeat(1, 3, 1, 1)
    return input


def get_transforms(multi_view=False, augmentations=True):

    """
    Build the data pipeline from path and integer label to tensor input for the model
    """

    transforms = []

    # make np array from label int
    transforms.append(label_to_np)
    # set up the dicom reader
    transforms.append(LoadImaged(keys=["image"], reader=ITKReader, image_only=True))
    # move the channel axis from last to first position
    transforms.append(move_channels_first)
    # resize to 227x277 (squeezenet default input size)
    transforms.append(Resized(keys=["image"], spatial_size=[227, 227]))
    # perform contrast-limited adaptive histogram equalization
    transforms.append(
        Clahed(
            keys=["image"],
            kernel_size=None,
            contrast_clip_limit=0.015,
            nbins=256,
            dtype=np.float32,
        )
    )
    # normalize the image based on the 5th and 95th percentile of the histogram
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
    # random augmentations
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
    # convert to pytorch tensor
    transforms.append(ToTensord(keys=["image", "label"], dtype=torch.float))
    # repeat the single channel image 3 times to make it RGB
    transforms.append(grey_to_rgb_mv if multi_view else grey_to_rgb_sv)

    return Compose(transforms)
