import os

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

import torchvision.models as torchmodels

from utils import (
    positive_metric_cmp_fn,
    stopping_fn_from_positive_metric,
    setup_device,
    seed_everything,
)
from transforms import get_train_transforms, get_val_transforms

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)




class PSCTrainer:

    def __init__(self) -> None:
        
        seed_everything(1234)

        # list of 100 dummy paths
        self.path_list = [f"path_{i}" for i in range(100)]

    def train_svcc(self):

        train_dataset = Dataset(self.path_list, transform=get_train_transforms(multi_view=False))
        train_dataloader = DataLoader(train_dataset, batch_size=14, shuffle=True, num_workers=0)

        val_dataset = Dataset(self.path_list, transform=get_val_transforms(multi_view=False))
        val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0)

        model = torchmodels.resnet18(num_classes=1)

        acc_list = self.train(model, train_dataloader, val_dataloader, max_epochs=3)

        print(acc_list)

    def train_mvcnn(self):

        train_dataset = Dataset(self.path_list, transform=get_train_transforms(multi_view=True))
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

        val_dataset = Dataset(self.path_list, transform=get_val_transforms(multi_view=True))
        val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0)

        model = torchmodels.resnet18(num_classes=1)

        # TODO: load weights from single model here

        acc_list = self.train(model, train_dataloader, val_dataloader, max_epochs=3)

        print(acc_list)


    def train(self, model, train_dataloader, val_dataloader, max_epochs=3):

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
            max_epochs=max_epochs,
            train_data_loader=train_dataloader,
            network=model,
            optimizer=optimizer,
            loss_function=loss,
            key_train_metric={"train_acc": train_acc},
            additional_metrics={"train_avg_loss": train_avg_loss},
            metric_cmp_fn=positive_metric_cmp_fn,
            decollate=False,
        )

        # engine.state.output = dict with keys ["image"] ["label"] and ["pred"]
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=val_dataloader,
            network=model,
            key_val_metric={"val_acc": val_acc},
            additional_metrics={"val_avg_loss": val_avg_loss},
            metric_cmp_fn=positive_metric_cmp_fn,
            decollate=False,
        )

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

        acc_list = []

        @evaluator.on(Events.EPOCH_COMPLETED(every=1))
        def store_acc_to_list(engine):
            acc_list.append(engine.state.metrics["val_acc"])

        trainer.run()

        return acc_list


if __name__ == "__main__":


    psc_trainer = PSCTrainer()

    psc_trainer.train_svcc()