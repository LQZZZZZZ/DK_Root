import argparse
import os
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from config_files.dk_root_Configs import Config as Configs
from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import base_Model
from trainer.trainer import Trainer, gen_pseudo_labels
from utils import _logger, copy_Files, set_requires_grad


def _append_tag_to_checkpoint(path: str, tag: Optional[str]) -> str:
    if not path or not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _log_dir_name(mode: str, seed: int, ne: int, nr: int, noise_ratio: float = 0.0) -> str:
    """Build a stable experiment directory name for checkpoints and logs."""
    base = f"{mode}_seed_{seed}"
    ne_suffix = None if (ne is None or ne <= 0) else ne
    nr_suffix = None if (nr is None or nr <= 0) else nr
    noise_suffix = None if (noise_ratio is None or noise_ratio <= 0) else f"noise{int(noise_ratio * 100)}"
    if mode in ["self_supervised", "SupCon"]:
        parts = [base]
        if nr_suffix is not None:
            parts.append(f"Nr{nr_suffix}")
        if noise_suffix is not None:
            parts.append(noise_suffix)
        return "_".join(parts)
    if mode in ["ft_2p", "train_linear_2p"]:
        parts = [base]
        if ne_suffix is not None:
            parts.append(f"Ne{ne_suffix}")
        if nr_suffix is not None:
            parts.append(f"Nr{nr_suffix}")
        if noise_suffix is not None:
            parts.append(noise_suffix)
        return "_".join(parts)
    if "supervised" in mode or any(marker in mode for marker in ["_2p", "_5p", "_10p", "_50p", "_75p"]):
        return base if ne_suffix is None else f"{base}_Ne{ne_suffix}"
    return base if nr_suffix is None else f"{base}_Nr{nr_suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DK-Root training entry point")
    home_dir = os.getcwd()
    parser.add_argument("--experiment_description", default="dk_root", type=str, help="Experiment group name")
    parser.add_argument("--run_description", default="quickstart", type=str, help="Run name")
    parser.add_argument("--seed", default=58, type=int, help="Random seed")
    parser.add_argument("--training_mode", default="supervised", type=str, help="Training mode")
    parser.add_argument("--selected_dataset", default=".", type=str, help="Dataset subdirectory under --data_path")
    parser.add_argument("--data_path", default=os.path.join(home_dir, "dataloader", "data_example"), type=str, help="Dataset root")
    parser.add_argument("--logs_save_dir", default=os.path.join(home_dir, "experiments_logs"), type=str, help="Log directory")
    parser.add_argument("--device", default="cpu", type=str, help="cpu, cuda, or cuda:N")
    parser.add_argument("--home_path", default=home_dir, type=str, help="Project home directory")
    parser.add_argument("--num_epoch", default=None, type=int, help="Override training epochs")
    parser.add_argument("--batch_size", default=None, type=int, help="Override batch size")
    parser.add_argument("--num_expert_samples", default=-1, type=int, help="Number of expert-labeled samples; -1 means all")
    parser.add_argument("--num_rule_samples", default=-1, type=int, help="Number of rule-labeled samples; -1 means all")
    parser.add_argument("--noise_ratio", default=0.0, type=float, help="Fraction of rule labels to randomly flip")
    parser.add_argument("--rule_data_path", default=None, type=str, help="Directory containing rule-labeled train.pt")
    parser.add_argument("--aug_method", default=None, type=str, help="Augmentation method")
    parser.add_argument("--diffusion_timesteps", default=None, type=int, help="Override diffusion timesteps")
    parser.add_argument("--diffusion_ddim_steps", default=None, type=int, help="Override DDIM sampling steps")
    parser.add_argument("--diffusion_num_epochs", default=None, type=int, help="Override diffusion training epochs")
    parser.add_argument("--diffusion_lr", default=None, type=float, help="Override diffusion learning rate")
    parser.add_argument("--diffusion_model_tag", default=None, type=str, help="Optional checkpoint suffix")
    parser.add_argument("--diffusion_weak_high_ratio", default=None, type=float, help="Weak augmentation upper timestep ratio")
    parser.add_argument("--diffusion_strong_low_ratio", default=None, type=float, help="Strong augmentation lower timestep ratio")
    return parser.parse_args()


def main() -> None:
    start_time = datetime.now()
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    configs = Configs()
    configs.device = args.device
    if args.num_epoch is not None:
        configs.num_epoch = args.num_epoch
    if args.batch_size is not None:
        configs.batch_size = args.batch_size
        configs.unlabeled_batch_size = args.batch_size
    if args.training_mode in ["self_supervised", "SupCon"]:
        configs.batch_size = configs.unlabeled_batch_size
        configs.num_epoch = configs.unlabeled_num_epoch
        configs.lr = configs.unlabeled_lr
    if args.training_mode == "ft_2p":
        configs.lr *= 0.5
    if args.aug_method is not None:
        configs.aug_method = args.aug_method
    if args.diffusion_timesteps is not None:
        configs.Diffusion.timesteps = args.diffusion_timesteps
    if args.diffusion_ddim_steps is not None:
        configs.Diffusion.ddim_steps = args.diffusion_ddim_steps
    if args.diffusion_num_epochs is not None:
        configs.Diffusion.num_epochs = args.diffusion_num_epochs
    if args.diffusion_lr is not None:
        configs.Diffusion.lr = args.diffusion_lr
    if args.diffusion_weak_high_ratio is not None:
        configs.Diffusion.weak_high_ratio = args.diffusion_weak_high_ratio
    if args.diffusion_strong_low_ratio is not None:
        configs.Diffusion.strong_low_ratio = args.diffusion_strong_low_ratio
    configs.Diffusion.save_path_diffusion = _append_tag_to_checkpoint(
        configs.Diffusion.save_path_diffusion_template.format(seed=args.seed), args.diffusion_model_tag
    )
    configs.Diffusion.save_path_diffusion_uncond = _append_tag_to_checkpoint(
        configs.Diffusion.save_path_diffusion_uncond_template.format(seed=args.seed), args.diffusion_model_tag
    )
    configs.TimeGAN.save_path_timegan = configs.TimeGAN.save_path_timegan_template.format(seed=args.seed)
    configs.TimeGAN.save_path_timegan_uncond = configs.TimeGAN.save_path_timegan_uncond_template.format(seed=args.seed)

    seed_everything(args.seed)
    os.makedirs(args.logs_save_dir, exist_ok=True)
    experiment_log_dir = os.path.join(
        args.logs_save_dir,
        args.experiment_description,
        args.run_description,
        _log_dir_name(args.training_mode, args.seed, args.num_expert_samples, args.num_rule_samples, args.noise_ratio),
    )
    os.makedirs(experiment_log_dir, exist_ok=True)
    logger = _logger(os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"))
    logger.debug("=" * 45)
    logger.debug(f"Dataset: {args.selected_dataset}")
    logger.debug(f"Mode:    {args.training_mode}")
    logger.debug("=" * 45)

    data_path = os.path.normpath(os.path.join(args.data_path, args.selected_dataset))
    train_dl, valid_dl, test_dl = data_generator(
        data_path,
        configs,
        args.training_mode,
        args.seed,
        num_expert_samples=args.num_expert_samples,
        num_rule_samples=args.num_rule_samples,
        noise_ratio=args.noise_ratio,
        rule_data_path=args.rule_data_path,
    )
    logger.debug("Data loaded ...")

    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
    if "fine_tune" in args.training_mode or "ft_" in args.training_mode or "train_linear" in args.training_mode or "tl" in args.training_mode:
        pretrain_dir = _log_dir_name("self_supervised", args.seed, args.num_expert_samples, args.num_rule_samples, args.noise_ratio)
        load_from = os.path.join(args.logs_save_dir, args.experiment_description, args.run_description, pretrain_dir, "saved_models")
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device, weights_only=True)
        pretrained_dict = {k: v for k, v in chkpoint["model_state_dict"].items() if "logits" not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if "train_linear" in args.training_mode or "tl" in args.training_mode:
            set_requires_grad(model, pretrained_dict, requires_grad=False)
    if args.training_mode == "gen_pseudo_labels":
        gen_pseudo_labels(model, train_dl, device, data_path, args.training_mode)
        sys.exit(0)
    if args.training_mode == "random_init":
        model_dict = {k: v for k, v in model.state_dict().items() if "logits" not in k}
        set_requires_grad(model, model_dict, requires_grad=False)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(
        temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4
    )
    if args.training_mode in ["self_supervised", "SupCon"]:
        copy_Files(os.path.join(args.logs_save_dir, args.experiment_description, args.run_description), args.selected_dataset)

    Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
            logger, configs, experiment_log_dir, args.training_mode, args.seed)
    logger.debug(f"Training time is : {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
