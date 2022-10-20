import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events
from monai.data import Dataset, DataLoader
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    ValidationHandler,
    EarlyStopHandler,
    CheckpointSaver,
)

from src.utils import (
    acc_from_lists,
    positive_metric_cmp_fn,
    stopping_fn_from_positive_metric,
    setup_device,
    seed_everything,
)
from src.transforms import get_transforms
from src.models import SVCNN, MVCNN

import logging

""" For multiprocessing"""
os.environ["MKL_THREADING_LAYER"] = "GNU"
torch.multiprocessing.set_start_method("spawn", force=True)  # for n_workers > 0

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DeePSCTrainer:
    def __init__(
        self,
        seed: int = 42,
        num_views: int = 7,
        num_workers: int = 0,
        repeat_sample_dataset: int = 5,
        use_gpu: bool = True,
        cpkt_folder: str = "checkpoints",
        image_folder: str = "images",
    ) -> None:

        seed_everything(seed)

        self.num_views = num_views
        self.num_workers = num_workers
        self.cpkt_folder = cpkt_folder
        self.image_folder = image_folder
        self.target_device = [0] if use_gpu else None

        self.create_example_data(repeat=repeat_sample_dataset)

    def create_example_data(
        self, repeat=5, example_folders=["pat_0_PSC", "pat_1_PSC", "pat_2_CG"]
    ):
        """
        Create example single- and multi-view data for training, validation and test
        as lists containing a dict per sample with {"image": "image_path", "label": 0/1}
        """

        self.sv_data_train = []
        self.mv_data_train = []

        self.sv_data_val = []
        self.mv_data_val = []

        self.sv_data_test = []
        self.mv_data_test = []

        for _ in range(repeat):
            for pat_folder in example_folders:
                self.sv_data_train, self.mv_data_train = self._append_sample_dicts(
                    self.sv_data_train,
                    self.mv_data_train,
                    pat_folder=pat_folder,
                )
                self.sv_data_val, self.mv_data_val = self._append_sample_dicts(
                    self.sv_data_val,
                    self.mv_data_val,
                    pat_folder=pat_folder,
                )
                self.sv_data_test, self.mv_data_test = self._append_sample_dicts(
                    self.sv_data_test,
                    self.mv_data_test,
                    pat_folder=pat_folder,
                )

    def train_ensemble(
        self, n_models=3, sv_lr=1e-4, mv_lr=1e-4, sv_epochs=5, mv_epochs=5, sv_bs=14, mv_bs=2
    ):

        self.ensemble_probs = []

        for i in range(n_models):

            log.info(f" ---------- Training ensemble model {i+1} of {n_models}")

            self.train_svcc(
                ckpt_name=f"svcnn_{i}.pt", lr=sv_lr, max_epochs=sv_epochs, batch_size=sv_bs
            )
            self.train_mvcnn(
                ckpt_name=f"mvcnn_{i}.pt",
                svcnn_cpkt_name=f"svcnn_{i}.pt",
                lr=mv_lr,
                max_epochs=mv_epochs,
                batch_size=mv_bs,
            )
            probs = self.test_mvcnn(ckpt_name=f"mvcnn_{i}.pt")
            self.ensemble_probs.append(probs)

        return self.ensemble_probs

    def deepsc_ensemble_prediction(self, ensemble_probs=None, threshold=0.5):

        if ensemble_probs is None:
            ensemble_probs = self.ensemble_probs

        df = pd.DataFrame(ensemble_probs).transpose()

        df["hce_prob"] = (df - threshold).apply(
            lambda x: max(x.min(), x.max(), key=abs), axis=1
        ) + threshold
        df["hce_pred"] = df["hce_prob"].apply(lambda x: 1 if x >= threshold else 0)
        df["label"] = [x["label"] for x in self.mv_data_test]

        deepsc_acc = acc_from_lists(df["label"], df["hce_pred"])
        log.info(f" ---------- DeePSC ensemble accuracy: {deepsc_acc}")

        return df, deepsc_acc

    def train_svcc(self, ckpt_name="svcnn.pt", lr=1e-4, max_epochs=1, batch_size=14):

        log.info(" ---------- Training SVCNN")

        train_dataset = Dataset(
            self.sv_data_train,
            transform=get_transforms(multi_view=False, augmentations=True),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_dataset = Dataset(
            self.sv_data_val,
            transform=get_transforms(multi_view=False, augmentations=False),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        model = SVCNN()

        acc_list = self._train(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_name=ckpt_name,
            lr=lr,
            max_epochs=max_epochs,
        )

        log.info(f" ---------- SVCNN validation accuracy over epochs: {acc_list}")

    def train_mvcnn(
        self, ckpt_name="mvcnn.pt", svcnn_cpkt_name="svcnn.pt", lr=1e-4, max_epochs=1, batch_size=2
    ):

        log.info(" ---------- Training MVCNN")

        train_dataset = Dataset(
            self.mv_data_train,
            transform=get_transforms(multi_view=True, augmentations=True),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_dataset = Dataset(
            self.mv_data_val,
            transform=get_transforms(multi_view=True, augmentations=False),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        single_model = SVCNN()

        if svcnn_cpkt_name:
            single_model.load_state_dict(
                torch.load(os.path.join(self.cpkt_folder, svcnn_cpkt_name))
            )
            log.info(f" ---------- Restored weights from SVCNN checkpoint: {svcnn_cpkt_name}")

        model = MVCNN(single_model=single_model, num_views=self.num_views)

        acc_list = self._train(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_name=ckpt_name,
            lr=lr,
            max_epochs=max_epochs,
        )

        log.info(f" ---------- MVCNN validation accuracy over epochs: {acc_list}")

    def test_mvcnn(self, ckpt_name="mvcnn.pt"):

        log.info(" ---------- Testing MVCNN")

        test_dataset = Dataset(
            self.mv_data_test,
            transform=get_transforms(multi_view=True, augmentations=False),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

        single_model = SVCNN()
        model = MVCNN(single_model=single_model, num_views=7)

        if ckpt_name:
            model.load_state_dict(torch.load(os.path.join(self.cpkt_folder, ckpt_name)))
            log.info(f" ---------- Restored weights from MVCNN checkpoint: {ckpt_name}")

        acc_list, probs_list = self._test(model, test_dataloader)

        log.info(f" ---------- MVCNN test accuracy: {acc_list}")

        return probs_list

    def _train(
        self, model, train_dataloader, val_dataloader, ckpt_name="cpkt.pt", lr=1e-4, max_epochs=3
    ):

        model, device = setup_device(model, target_devices=self.target_device)

        optimizer = AdamW([{"params": model.parameters()}], lr=lr)

        loss = nn.BCEWithLogitsLoss()

        train_acc = Accuracy(
            output_transform=lambda x: (
                torch.sigmoid(x["pred"].float()).round(),
                x["label"],
            ),
            device=device,
        )
        val_acc = Accuracy(
            output_transform=lambda x: (
                torch.sigmoid(x["pred"].float()).round(),
                x["label"],
            ),
            device=device,
        )
        train_avg_loss = Loss(
            loss,
            output_transform=lambda output: (output["pred"], output["label"]),
            device=device,
        )
        val_avg_loss = Loss(
            loss,
            output_transform=lambda output: (output["pred"], output["label"]),
            device=device,
        )

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
            name="val_evaluator",
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

    def _test(self, model, test_dataloader):

        model, device = setup_device(model, target_devices=self.target_device)

        loss = nn.BCEWithLogitsLoss()

        test_acc = Accuracy(
            output_transform=lambda x: (
                torch.sigmoid(x["pred"].float()).round(),
                x["label"],
            ),
            device=device,
        )
        test_avg_loss = Loss(
            loss,
            output_transform=lambda output: (output["pred"], output["label"]),
            device=device,
        )

        # engine.state.output = dict with keys ["image"] ["label"] and ["pred"]
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=test_dataloader,
            network=model,
            key_val_metric={"test_acc": test_acc},
            additional_metrics={"test_avg_loss": test_avg_loss},
            metric_cmp_fn=positive_metric_cmp_fn,
            decollate=False,
        )

        StatsHandler(
            output_transform=lambda x: None,
            name="test_evaluator",
        ).attach(evaluator)

        acc_list = []
        probs_list = []

        @evaluator.on(Events.EPOCH_COMPLETED(every=1))
        def store_acc_to_list(engine):
            acc_list.append(engine.state.metrics["test_acc"])

        @evaluator.on(Events.ITERATION_COMPLETED(every=1))
        def store_preds_to_list(engine):
            for pred in engine.state.output["pred"]:
                probs_list.append(torch.sigmoid(pred).cpu().detach().item())

        evaluator.run()

        return acc_list, probs_list

    def _append_sample_dicts(self, sv_list, mv_list, pat_folder="pat_0_PSC"):

        for i in range(self.num_views):
            sv_sample = {}
            sv_sample["image"] = f"{self.image_folder}/{pat_folder}/view_{i}.dcm"
            sv_sample["label"] = 1 if "PSC" in pat_folder else 0
            sv_list.append(sv_sample)
        mv_sample = {}
        mv_sample["image"] = f"{self.image_folder}/{pat_folder}"
        mv_sample["label"] = 1 if "PSC" in pat_folder else 0
        mv_list.append(mv_sample)

        return sv_list, mv_list
