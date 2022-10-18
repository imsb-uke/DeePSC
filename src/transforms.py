import torch
from monai.transforms import Compose, RandAffineDict

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

def grey_to_rgb_sv(input):
    # repeat the single channel image 3 times to make it RGB
    input["image"] = input["image"].repeat(3, 1, 1)
    # shape is now (3, 256, 256)
    return input

def grey_to_rgb_mv(input):
    # add a channel axis to the image (before the 7 views were stacked in the channel axis)
    # repeat the single channel image 3 times to make it RGB
    input["image"] = input["image"].unsqueeze(1).repeat(1, 3, 1, 1)
    # shape is now (7, 3, 256, 256)
    return input

def cast_label_type(input):
    input["label"] = input["label"].float()
    return input

def get_train_transforms(multi_view=False):

    img_loader = load_image_mv if multi_view else load_image_sv
    grey_to_rgb = grey_to_rgb_mv if multi_view else grey_to_rgb_sv


    # TODO: add Histo Eq and all augmentations

    affine_trans = RandAffineDict(
        keys=["image"],
        prob=0.5,
        rotate_range=0.2,
        shear_range=0.2,
        translate_range=0.2,
        scale_range=0.2,
    )

    transforms = Compose([img_loader, cast_label_type, affine_trans, grey_to_rgb])

    return transforms

def get_val_transforms(multi_view=False):

    img_loader = load_image_mv if multi_view else load_image_sv
    grey_to_rgb = grey_to_rgb_mv if multi_view else grey_to_rgb_sv

    transforms = Compose([img_loader, cast_label_type, grey_to_rgb])

    return transforms
