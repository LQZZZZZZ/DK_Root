import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import DataTransform, DataTransform_diffusion, DataTransform_diffusion_uncond


def _ensure_1d_labels(labels):
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if len(labels.shape) == 2:
        labels = labels.squeeze(1)
    return labels.long()


def _subsample_indices_stratified(labels: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Sample indices as evenly as possible across classes."""
    labels = _ensure_1d_labels(labels).cpu()
    total = labels.numel()
    if n <= 0 or n >= total:
        return torch.arange(total)
    rng = np.random.RandomState(seed)
    unique_classes = sorted(int(value) for value in labels.unique().tolist())
    class_order = unique_classes.copy()
    rng.shuffle(class_order)
    indices_by_class = {}
    for class_id in unique_classes:
        indices = (labels == class_id).nonzero(as_tuple=False).view(-1).numpy()
        rng.shuffle(indices)
        indices_by_class[class_id] = indices
    if n < len(unique_classes):
        picked = [indices_by_class[class_id][0] for class_id in class_order[:n] if len(indices_by_class[class_id]) > 0]
        return torch.tensor(picked, dtype=torch.long)
    base = n // len(unique_classes)
    remainder = n % len(unique_classes)
    picked = []
    remaining_capacity = {}
    for class_id in unique_classes:
        take = min(base, len(indices_by_class[class_id]))
        picked.extend(indices_by_class[class_id][:take].tolist())
        remaining_capacity[class_id] = max(0, len(indices_by_class[class_id]) - take)
        indices_by_class[class_id] = indices_by_class[class_id][take:]
    if remainder > 0:
        ordered_classes = sorted(unique_classes, key=lambda class_id: (remaining_capacity[class_id], class_order.index(class_id)), reverse=True)
        for class_id in ordered_classes:
            if remainder <= 0:
                break
            if len(indices_by_class[class_id]) == 0:
                continue
            picked.append(indices_by_class[class_id][0])
            indices_by_class[class_id] = indices_by_class[class_id][1:]
            remainder -= 1
    if len(picked) < n:
        remaining = []
        for class_id in unique_classes:
            remaining.extend(indices_by_class[class_id].tolist())
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])
    return torch.tensor(picked[:n], dtype=torch.long)


def _subsample_dataset_dict(dataset: dict, n: int, seed: int, stratified: bool = True) -> dict:
    """Subsample a dataset dictionary while preserving optional metadata."""
    if n is None or n <= 0:
        dataset["labels"] = _ensure_1d_labels(dataset["labels"])
        return dataset
    samples = dataset["samples"]
    labels = _ensure_1d_labels(dataset["labels"])
    if n >= labels.shape[0]:
        dataset["labels"] = labels
        return dataset
    if stratified:
        indices = _subsample_indices_stratified(labels, n, seed)
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(labels.shape[0], generator=generator)[:n]
    out = {"samples": samples[indices], "labels": labels[indices]}
    if "window_num" in dataset:
        window_num = dataset["window_num"]
        if isinstance(window_num, torch.Tensor):
            out["window_num"] = window_num[indices]
        elif isinstance(window_num, np.ndarray):
            out["window_num"] = window_num[indices.cpu().numpy()]
        else:
            out["window_num"] = [window_num[index] for index in indices.cpu().tolist()]
    for key in ["fields", "label_classes"]:
        if key in dataset:
            out[key] = dataset[key]
    return out


def _inject_label_noise(dataset: dict, noise_ratio: float, num_classes: int, seed: int) -> dict:
    """Inject symmetric label noise into a dataset dictionary."""
    if noise_ratio is None or noise_ratio <= 0:
        return dataset
    labels = dataset["labels"].clone()
    rng = np.random.RandomState(seed)
    n_flip = int(labels.shape[0] * noise_ratio)
    flip_indices = rng.choice(labels.shape[0], size=n_flip, replace=False)
    for index in flip_indices:
        old_label = labels[index].item()
        candidates = [class_id for class_id in range(num_classes) if class_id != old_label]
        labels[index] = rng.choice(candidates)
    dataset["labels"] = labels
    return dataset


class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        x_train = dataset["samples"]
        y_train = _ensure_1d_labels(dataset["labels"])
        self.window_num = dataset.get("window_num", torch.arange(len(x_train)))
        if len(x_train.shape) < 3:
            x_train = x_train.unsqueeze(2)
        self.x_data = torch.from_numpy(x_train) if isinstance(x_train, np.ndarray) else x_train
        self.y_data = torch.from_numpy(y_train) if isinstance(y_train, np.ndarray) else y_train
        self.len = self.x_data.shape[0]
        if torch.unique(self.y_data).numel() == config.num_classes and config.aug_method == "diffusion":
            self.aug1, self.aug2 = DataTransform_diffusion(self.x_data, self.y_data, config)
        elif torch.unique(self.y_data).numel() == config.num_classes and config.aug_method == "diffusion_uncond":
            self.aug1, self.aug2 = DataTransform_diffusion_uncond(self.x_data, self.y_data, config)
        elif torch.unique(self.y_data).numel() == config.num_classes and config.aug_method == "timegan":
            from VariousAugMethod4ComExper.timegan_augmentation import DataTransform_timegan
            self.aug1, self.aug2 = DataTransform_timegan(self.x_data, self.y_data, config)
        elif torch.unique(self.y_data).numel() == config.num_classes and config.aug_method == "timegan_uncond":
            from VariousAugMethod4ComExper.timegan_uncond_augmentation import DataTransform_timegan_uncond
            self.aug1, self.aug2 = DataTransform_timegan_uncond(self.x_data, self.y_data, config)
        else:
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode in ["self_supervised", "SupCon", "supervised", "supervised_after_cgan", "supervised_aug", "supervised_full_data"]:
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index], self.window_num[index]
        return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index], self.window_num[index]

    def __len__(self):
        return self.len


def min_max_normalize(data, min_vals, max_vals, epsilon=1e-8):
    """Apply min-max normalization with precomputed statistics."""
    return (data - min_vals) / (max_vals - min_vals + epsilon)


def _load_dataset(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def data_generator(data_path, configs, training_mode, seed, num_expert_samples=-1, num_rule_samples=-1, noise_ratio=0.0, rule_data_path=None):
    batch_size = configs.batch_size
    rule_data_path = data_path if rule_data_path is None else rule_data_path
    if "_2p" in training_mode or training_mode in ["supervised", "supervised_after_cgan"] or "supervised_aug" in training_mode:
        train_dataset = _load_dataset(os.path.join(data_path, "train_2p_labeled.pt"))
        train_dataset = _subsample_dataset_dict(train_dataset, num_expert_samples, seed, stratified=True)
    elif "_5p" in training_mode:
        train_dataset = _load_dataset(os.path.join(data_path, "train_5perc.pt"))
    elif "_10p" in training_mode:
        train_dataset = _load_dataset(os.path.join(data_path, "train_10perc.pt"))
    elif "_50p" in training_mode:
        train_dataset = _load_dataset(os.path.join(data_path, "train_50perc.pt"))
    elif "_75p" in training_mode:
        train_dataset = _load_dataset(os.path.join(data_path, "train_75perc.pt"))
    elif training_mode in ["self_supervised", "SupCon"]:
        train_dataset = _load_dataset(os.path.join(rule_data_path, "train.pt"))
        train_dataset = _subsample_dataset_dict(train_dataset, num_rule_samples, seed, stratified=True)
        train_dataset = _inject_label_noise(train_dataset, noise_ratio, configs.num_classes, seed)
    elif training_mode == "supervised_full_data":
        expert_dataset = _subsample_dataset_dict(_load_dataset(os.path.join(data_path, "train_2p_labeled.pt")), num_expert_samples, seed, stratified=True)
        rule_dataset = _subsample_dataset_dict(_load_dataset(os.path.join(rule_data_path, "train.pt")), num_rule_samples, seed, stratified=True)
        train_dataset = {
            "samples": torch.cat([expert_dataset["samples"].cpu(), rule_dataset["samples"].cpu()], dim=0),
            "labels": torch.cat([expert_dataset["labels"].cpu(), rule_dataset["labels"].cpu()], dim=0),
        }
    else:
        train_dataset = _load_dataset(os.path.join(rule_data_path, "train.pt"))
        train_dataset = _subsample_dataset_dict(train_dataset, num_rule_samples, seed, stratified=True)

    valid_path = os.path.join(data_path, "val_added.pt") if os.path.exists(os.path.join(data_path, "val_added.pt")) else os.path.join(data_path, "val.pt")
    valid_dataset = _load_dataset(valid_path)
    test_dataset = _load_dataset(os.path.join(data_path, "val.pt"))

    if configs.use_normalization and training_mode not in ["SupCon"]:
        train_samples = train_dataset["samples"].to(configs.device)
        valid_samples = valid_dataset["samples"].to(configs.device)
        test_samples = test_dataset["samples"].to(configs.device)
        min_vals = train_samples.amin(dim=(0, 2), keepdim=True)
        max_vals = train_samples.amax(dim=(0, 2), keepdim=True)
        train_dataset["samples"] = min_max_normalize(train_samples, min_vals, max_vals).cpu()
        valid_dataset["samples"] = min_max_normalize(valid_samples, min_vals, max_vals).cpu()
        test_dataset["samples"] = min_max_normalize(test_samples, min_vals, max_vals).cpu()

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)
    train_len = len(train_dataset)
    drop_last = configs.drop_last
    if train_len < batch_size:
        batch_size = max(1, train_len)
        drop_last = False
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=False, num_workers=0, shuffle=True)
    return train_loader, valid_loader, test_loader
