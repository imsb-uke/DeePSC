import os
import torch

from src.trainer import DeePSCTrainer

""" For multiprocessing"""
os.environ["MKL_THREADING_LAYER"] = "GNU"
torch.multiprocessing.set_start_method("spawn", force=True)  # for n_workers > 0


def main():

    psc_trainer = DeePSCTrainer(
        num_views=7,
        num_workers=2,
        repeat_sample_dataset=5,
        use_gpu=True,
    )

    psc_trainer.train_ensemble(
        n_models=3, sv_lr=1e-4, mv_lr=1e-4, sv_epochs=5, mv_epochs=5, sv_bs=14, mv_bs=2
    )

    preds_df, deepsc_acc = psc_trainer.deepsc_ensemble_prediction()


if __name__ == "__main__":
    main()
