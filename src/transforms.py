import torch
from monai.transforms import Compose, RandAffineDict

def load_single_view_image(path):
    # dummy function that loads image from a path
    _ = path
    image = torch.rand(1, 512, 512)
    label = torch.randint(0, 1, (1,))
    return {"image": image, "label": label}

def load_multi_view_image(path, num_views=7):
    # dummy function that loads image from a path
    _ = path
    image = torch.rand(num_views, 512, 512)
    label = torch.randint(0, 1, (1,))
    return {"image": image, "label": label}

def cast_label_type(input):
    input["label"] = input["label"].float()
    return input

def grey_to_rgb(input):
    input["image"] = input["image"].repeat(3, 1, 1)
    return input

def get_train_transforms(multi_view=False):

    img_loader = load_multi_view_image if multi_view else load_single_view_image

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

    img_loader = load_multi_view_image if multi_view else load_single_view_image

    transforms = Compose([img_loader, cast_label_type, grey_to_rgb])

    return transforms
