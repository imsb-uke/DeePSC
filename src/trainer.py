import os

""" For multiprocessing"""
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

""" For multiprocessing"""
torch.multiprocessing.set_start_method("spawn", force=True)  # for n_workers > 0

from copy import copy
import torch.nn as nn
from torch.optim import AdamW
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events

from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import StatsHandler, ValidationHandler, EarlyStopHandler, CheckpointSaver
from monai.data import Dataset, DataLoader

from utils import (
    positive_metric_cmp_fn,
    stopping_fn_from_positive_metric,
    setup_device,
    seed_everything,
)
from transforms import get_train_transforms, get_val_transforms
from models import SVCNN, MVCNN

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PSCTrainer:

    def __init__(self) -> None:
        
        seed_everything(1234)

        self.cpkt_folder = "checkpoints"
        self.num_views = 7
        self.patient_batch_size = 2
        self.num_workers = 0

        self.create_data(repeat=5)

    def create_data(self, repeat=5):

        self.sv_data_train = []
        self.mv_data_train = []

        for _ in range(repeat):
            for i in range(self.num_views):
                sv_sample = {}
                sv_sample["image"] = f"images/pat_0/view_{i}.dcm"
                sv_sample["label"] = 1
                self.sv_data_train.append(sv_sample)
            mv_sample = {}
            mv_sample["image"] = f"images/pat_0"
            mv_sample["label"] = 1
            self.mv_data_train.append(mv_sample)

        self.sv_data_val = []
        self.mv_data_val = []

        for _ in range(repeat):
            for i in range(self.num_views):
                sv_sample = {}
                sv_sample["image"] = f"images/pat_0/view_{i}.dcm"
                sv_sample["label"] = 1
                self.sv_data_val.append(sv_sample)
            mv_sample = {}
            mv_sample["image"] = f"images/pat_0"
            mv_sample["label"] = 1
            self.mv_data_val.append(mv_sample)

    def train_svcc(self, ckpt_name="svcnn.pt", max_epochs=1):

        train_dataset = Dataset(self.sv_data_train, transform=get_train_transforms(multi_view=False))
        train_dataloader = DataLoader(train_dataset, batch_size=self.num_views*self.patient_batch_size, shuffle=True, num_workers=self.num_workers)

        val_dataset = Dataset(self.sv_data_val, transform=get_val_transforms(multi_view=False))
        val_dataloader = DataLoader(val_dataset, batch_size=self.num_views*self.patient_batch_size, shuffle=False, num_workers=self.num_workers)

        model = SVCNN()

        acc_list = self.train(model, train_dataloader, val_dataloader, ckpt_name=ckpt_name, max_epochs=max_epochs)

        print(acc_list)

    def train_mvcnn(self, ckpt_name="mvcnn.pt", svcnn_cpkt_name="svcnn.pt", max_epochs=1):

        train_dataset = Dataset(self.mv_data_train, transform=get_train_transforms(multi_view=True))
        train_dataloader = DataLoader(train_dataset, batch_size=self.patient_batch_size, shuffle=True, num_workers=self.num_workers)

        val_dataset = Dataset(self.mv_data_val, transform=get_val_transforms(multi_view=True))
        val_dataloader = DataLoader(val_dataset, batch_size=self.patient_batch_size, shuffle=False, num_workers=self.num_workers)

        single_model = SVCNN()
        single_model.load_state_dict(torch.load(os.path.join(self.cpkt_folder, svcnn_cpkt_name)))

        model = MVCNN(single_model = single_model, num_views=7)

        acc_list = self.train(model, train_dataloader, val_dataloader, ckpt_name=ckpt_name, max_epochs=max_epochs)

        print(acc_list)

    def train(self, model, train_dataloader, val_dataloader, ckpt_name="cpkt.pt", max_epochs=3):

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

        CheckpointSaver(
            save_dir=self.cpkt_folder,
            save_dict={"model": model},
            save_final=False,
            final_filename="_.pt",
            save_key_metric=True,
            key_metric_filename=ckpt_name,
            key_metric_negative_sign=False,
        ).attach(evaluator)

        acc_list = []

        @evaluator.on(Events.EPOCH_COMPLETED(every=1))
        def store_acc_to_list(engine):
            acc_list.append(engine.state.metrics["val_acc"])

        trainer.run()

        return acc_list

    def test(self):
        pass

    def deepsc_ensemble_test(self):
        pass


if __name__ == "__main__":

    psc_trainer = PSCTrainer()

    psc_trainer.train_svcc()
    psc_trainer.train_mvcnn()