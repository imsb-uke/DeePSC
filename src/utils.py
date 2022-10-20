import os
import torch
import torch.nn as nn
from typing import List, Tuple
from monai.utils import set_determinism as monai_set_determinism
from ignite.engine import Engine


def acc_from_lists(true_list, pred_list):
    """
    Compute accuracy from a list of labels and a list of predictions {0,1}
    """
    return sum(1 for x, y in zip(true_list, pred_list) if x == y) / len(true_list)


def positive_metric_cmp_fn(current_metric: float, prev_best: float) -> bool:

    return current_metric > prev_best


def negative_metric_cmp_fn(current_metric: float, prev_best: float) -> bool:

    # Workaround for -1 default value in engine.state.best_metric
    if isinstance(prev_best, int) and prev_best == -1:
        prev_best = 99999

    return current_metric < prev_best


def stopping_fn_from_negative_metric(metric_name: str):
    """
    Returns a stopping function for ignite.handlers.EarlyStopping using the given metric name.
    """

    def stopping_fn(engine: Engine):
        return -engine.state.metrics[metric_name]

    return stopping_fn


def stopping_fn_from_positive_metric(metric_name: str):
    """
    Returns a stopping function for ignite.handlers.EarlyStopping using the given metric name.
    """

    def stopping_fn(engine: Engine):
        return engine.state.metrics[metric_name]

    return stopping_fn


def seed_everything(seed: int) -> None:
    monai_set_determinism(seed=seed, additional_settings=None)
    if seed:
        os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device(model: nn.Module, target_devices: List[int]) -> Tuple[torch.device, List[int]]:
    """
    setup GPU device if available, move model into configured device
    """
    available_devices = list(range(torch.cuda.device_count()))

    if not available_devices:
        print("There's no GPU available on this machine. Training will be performed on CPU.")
        device = torch.device("cpu")
        model = model.to(device)
        return model, device

    if not target_devices:
        print("No GPU selected. Training will be performed on CPU.")
        device = torch.device("cpu")
        model = model.to(device)
        return model, device

    max_target_gpu = max(target_devices)
    max_available_gpu = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg = (
            f"Configuration requests GPU #{max_target_gpu} but only {max_available_gpu} "
            "available. Check the configuration and try again."
        )
        raise Exception(msg)

    print(f"Using devices {target_devices} of available devices {available_devices}")
    device = torch.device(f"cuda:{target_devices[0]}")
    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices).to(device)
    else:
        model = model.to(device)
    return model, device
