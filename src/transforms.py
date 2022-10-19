import torch
import numpy as np
from monai.transforms import Compose, LoadImaged, RandAffined, ToTensord
from monai.data.image_reader import ITKReader

# TODO: add dummy tiffs and use monai loader!

def load_image_sv(path):
    # dummy function that loads greyscale image from a path
    _ = path
    image = torch.rand(1, 256, 256)
    label = torch.randint(0, 1, (1,))
    return {"image": image, "label": label}

def load_image_mv(path, num_views=7):
    # dummy function that loads num_views greyscale images from a path
    # images are stacked in the channel axis
    _ = path
    image = torch.rand(num_views, 256, 256)
    label = torch.randint(0, 1, (1,))
    return {"image": image, "label": label}

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

def cast_label_type(input, key="label"):
    input[key] = torch.tensor(input[key], dtype=torch.float)
    return input

def move_channels_first(input, key="image"):
    # move the channel axis to the first position
    input[key] = np.transpose(input[key], (2, 1, 0))
    return input   

def to_np(input, key="label"):
    input[key] = np.array([input[key]])
    return input

def get_train_transforms(multi_view=False):

    grey_to_rgb = grey_to_rgb_mv if multi_view else grey_to_rgb_sv

    img_loader = LoadImaged(keys=["image"], reader=ITKReader, image_only=True)

    # TODO: add normalization
    # TODO: add augmentations
    # TODO: add histogram equalization

    affine_trans = RandAffined(
        keys=["image"],
        prob=0.5,
        rotate_range=0.2,
        shear_range=0.2,
        translate_range=0.2,
        scale_range=0.2,
    )

    to_tensor = ToTensord(keys=["image", "label"], dtype=torch.float)

    transforms = Compose([to_np, img_loader, move_channels_first, affine_trans, to_tensor, grey_to_rgb])

    return transforms

def get_val_transforms(multi_view=False):

    grey_to_rgb = grey_to_rgb_mv if multi_view else grey_to_rgb_sv

    img_loader = LoadImaged(keys=["image"], reader=ITKReader, image_only=True)

    to_tensor = ToTensord(keys=["image", "label"], dtype=torch.float)

    transforms = Compose([to_np, img_loader, move_channels_first, to_tensor, grey_to_rgb])

    return transforms
