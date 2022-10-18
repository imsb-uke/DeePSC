import os

"Always set environ variables before importing the rest!"
""" For setting the number of threads in the background for numpy (MKL/BLAS etc)"""
# os.environ["OPENBLAS_NUM_THREADS"]='1'
# os.environ["NUMBA_NUM_THREADS"]='1'
os.environ["OMP_NUM_THREADS"] = "1"

""" For multiprocessing"""
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

""" For multiprocessing"""
torch.multiprocessing.set_start_method("spawn", force=True)  # for n_workers > 0

import torch.nn as nn
from torch.optim import AdamW
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events

from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import StatsHandler, ValidationHandler, EarlyStopHandler
from monai.data import Dataset, DataLoader
from monai.utils.misc import first
from monai.transforms import Compose, RandAffineDict

import torchvision.models as torchmodels

from src.utils import (
    positive_metric_cmp_fn,
    stopping_fn_from_positive_metric,
    setup_device,
    seed_everything,
)

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# list of 100 dummy paths
path_list = [f"path_{i}" for i in range(100)]


def load_image(path):
    # dummy function that loads image from a path
    _ = path
    image = torch.rand(3, 512, 512)
    label = torch.randint(0, 1, (1,))
    return {"image": image, "label": label}


def cast_types(input):
    input["label"] = input["label"].float()
    return input


affine_trans = RandAffineDict(
    keys=["image"],
    prob=0.5,
    rotate_range=0.2,
    shear_range=0.2,
    translate_range=0.2,
    scale_range=0.2,
)

# create a pytorch dataset from the path lists, where all preprocessing steps are in "transforms"
train_dataset = Dataset(path_list, transform=Compose([load_image, cast_types, affine_trans]))
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

val_dataset = Dataset(path_list, transform=Compose([load_image, cast_types]))
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0)

model = torchmodels.resnet18(num_classes=1)
model, device = setup_device(model, target_devices=[0])

optimizer = AdamW([{"params": model.parameters()}])

loss = nn.BCEWithLogitsLoss()

train_acc = Accuracy(output_transform=lambda x: (torch.sigmoid(x["pred"].float()).round(), x["label"]), device=device)
val_acc = Accuracy(output_transform=lambda x: (torch.sigmoid(x["pred"].float()).round(), x["label"]), device=device)
train_avg_loss = Loss(loss, output_transform=lambda output: (output["pred"], output["label"]), device=device)
val_avg_loss = Loss(loss, output_transform=lambda output: (output["pred"], output["label"]), device=device)

# engine.state.output = dict with keys ["image"] ["label"] ["pred"] and ["loss"]
trainer = SupervisedTrainer(
    device=device,
    max_epochs=10,
    train_data_loader=train_dataloader,
    network=model,
    optimizer=optimizer,
    loss_function=loss,
    key_train_metric={"train_acc": train_acc},
    additional_metrics={"train_avg_loss": train_avg_loss},
    metric_cmp_fn=positive_metric_cmp_fn,  # careful when changing key metric !
    decollate=False,
)

# engine.state.output = dict with keys ["image"] ["label"] and ["pred"]
evaluator = SupervisedEvaluator(
    device=device,
    val_data_loader=val_dataloader,
    network=model,
    key_val_metric={"val_acc": val_acc},
    additional_metrics={"val_avg_loss": val_avg_loss},
    metric_cmp_fn=positive_metric_cmp_fn,  # careful when changing key metric !
    decollate=False,
)

# one way to attach handlers: use "attach" function if they have one

StatsHandler(
    output_transform=lambda output: {"iteration_loss": output["loss"]},
    name="trainer",
).attach(trainer)

StatsHandler(
    output_transform=lambda x: None,
    global_epoch_transform=lambda x: trainer.state.epoch,
    name="evaluator",
).attach(evaluator)

# run evaluator on every epoch
ValidationHandler(validator=evaluator, interval=1, epoch_level=True).attach(trainer)

EarlyStopHandler(
    patience=3,
    score_function=stopping_fn_from_positive_metric("val_acc"),
    trainer=trainer,
).attach(evaluator)


# another nice way to have custom handlers on events: use decorator

acc_list = []


@trainer.on(Events.EPOCH_COMPLETED(every=1))
def store_acc_to_list(engine):
    acc_list.append(trainer.state.metrics["train_acc"])


if __name__ == "__main__":

    seed_everything(1234)

    # print(first(train_dataloader))

    trainer.run()

    print(acc_list)
