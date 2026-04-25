import numpy as np
import torch


def DataTransform(sample, config):
    """Create weak and strong views using lightweight time-series augmentations."""
    sigma = getattr(config.augmentation, "jitter_scale_ratio_strong", 1.1)
    noise_ratio = getattr(config.augmentation, "noise_rate", 0.1)
    strong_aug = scaling(sample, sigma=sigma)
    weak_aug = adding_noise(sample, ratio=noise_ratio)
    return weak_aug.float(), strong_aug.float()


def adding_noise(x, ratio=0.1):
    """Add feature-wise Gaussian noise scaled by each channel standard deviation."""
    x_np = x.detach().cpu().numpy()
    std = np.std(x_np, axis=2, keepdims=True)
    noise = np.random.normal(loc=0.0, scale=1.0, size=x_np.shape)
    return torch.from_numpy(x_np + ratio * std * noise)


def jitter(x, sigma=0.8):
    """Apply random jittering to a tensor or array."""
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    return torch.from_numpy(x_np + np.random.normal(loc=0.0, scale=sigma, size=x_np.shape))


def scaling(x, sigma=1.1):
    """Scale each sample with a random temporal factor."""
    x_np = x.detach().cpu().numpy()
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x_np.shape[0], x_np.shape[2]))
    scaled_channels = []
    for channel_index in range(x_np.shape[1]):
        channel = x_np[:, channel_index, :]
        scaled_channels.append(np.multiply(channel, factor)[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate(scaled_channels, axis=1))


def permutation(x, max_segments=5, seg_mode="random"):
    """Randomly permute temporal segments."""
    original_steps = np.arange(x.shape[2])
    x_np = x.detach().cpu().numpy()
    num_segments = np.random.randint(1, max_segments, size=(x_np.shape[0]))
    ret = np.zeros_like(x_np)
    for sample_index, pattern in enumerate(x_np):
        if num_segments[sample_index] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x_np.shape[2] - 2, num_segments[sample_index] - 1, replace=False)
                split_points.sort()
                splits = np.split(original_steps, split_points)
            else:
                splits = np.array_split(original_steps, num_segments[sample_index])
            warp = np.concatenate([np.random.permutation(segment) for segment in splits]).ravel()
            ret[sample_index] = pattern[0, warp]
        else:
            ret[sample_index] = pattern
    return torch.from_numpy(ret)


def DataTransform_diffusion(sample, label, config):
    """Create weak and strong views with a pretrained conditional diffusion model."""
    from Diffusion_aug_main import DiffusionModel

    device = torch.device(config.device)
    generator = DiffusionModel(
        feature_dim=config.input_channels,
        length_seq=config.sequence_length,
        timesteps=config.Diffusion.timesteps,
        device=device,
    )
    generator.load_state_dict(torch.load(config.Diffusion.save_path_diffusion, map_location=device, weights_only=True))
    generator.eval()
    return _diffusion_views(generator, sample, label, config, device)


def DataTransform_diffusion_uncond(sample, label, config):
    """Create weak and strong views with a pretrained unconditional diffusion model."""
    from Diffusion_aug_main import DiffusionModel

    device = torch.device(config.device)
    generator = DiffusionModel(
        feature_dim=config.input_channels,
        length_seq=config.sequence_length,
        timesteps=config.Diffusion.timesteps,
        num_classes=0,
        device=device,
    )
    generator.load_state_dict(torch.load(config.Diffusion.save_path_diffusion_uncond, map_location=device, weights_only=True))
    generator.eval()
    return _diffusion_views(generator, sample, None, config, device)


def _diffusion_views(generator, sample, label, config, device):
    total_steps = generator.timesteps
    weak_high_ratio = getattr(config.Diffusion, "weak_high_ratio", 0.125)
    strong_low_ratio = getattr(config.Diffusion, "strong_low_ratio", 0.5)
    weak_high = max(1, int(total_steps * weak_high_ratio))
    strong_low = max(1, int(total_steps * strong_low_ratio))
    sample_dev = sample.to(device)
    label_dev = None if label is None else label.to(device)
    t_weak = torch.randint(low=0, high=weak_high, size=(sample.shape[0],), device=device)
    t_strong = torch.randint(low=strong_low, high=total_steps, size=(sample.shape[0],), device=device)
    ddim_steps = getattr(config.Diffusion, "ddim_steps", 1)
    diff_aug_strong, _ = generator.augment_sample(sample_dev, label_dev, t_strong, ddim_steps=ddim_steps)
    diff_aug_weak, _ = generator.augment_sample(sample_dev, label_dev, t_weak, ddim_steps=ddim_steps)
    return diff_aug_strong.cpu(), diff_aug_weak.cpu()
