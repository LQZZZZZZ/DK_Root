import logging
import os
import random
import sys
from shutil import copy

import numpy as np
import torch


def set_requires_grad(model, dict_, requires_grad=True):
    for name, parameter in model.named_parameters():
        if name in dict_:
            parameter.requires_grad = requires_grad


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(logger_name, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, data_type):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    files_to_copy = [
        ("main.py", "main.py"),
        (os.path.join("trainer", "trainer.py"), "trainer.py"),
        (os.path.join("config_files", "dk_root_Configs.py"), "dk_root_Configs.py"),
        (os.path.join("dataloader", "augmentations.py"), "augmentations.py"),
        (os.path.join("dataloader", "dataloader.py"), "dataloader.py"),
        (os.path.join("models", "model.py"), "model.py"),
        (os.path.join("models", "loss.py"), "loss.py"),
        (os.path.join("models", "TC.py"), "TC.py"),
    ]
    for source, target in files_to_copy:
        source_path = os.path.join(script_dir, source)
        if os.path.exists(source_path):
            copy(source_path, os.path.join(destination_dir, target))
