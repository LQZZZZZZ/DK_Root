import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config_files.dk_root_Configs import Config


class MyDataset(Dataset):
    def __init__(self, data_dict):
        self.samples = data_dict["samples"].float()
        self.labels = data_dict["labels"].long()

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, index):
        return {"sample": self.samples[index], "labels": self.labels[index]}


class UNet1D(nn.Module):
    """Small 1D U-Net denoiser for KPI sequences."""

    def __init__(self, in_channels=33, base_channels=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, base_channels, 3, padding=1), nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True))
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.BatchNorm1d(base_channels * 2), nn.ReLU(inplace=True), nn.Conv1d(base_channels * 2, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv1d(base_channels * 2, base_channels, 3, padding=1), nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv1d(base_channels, in_channels, 1)

    def forward(self, x):
        enc = self.enc1(x)
        hidden = self.enc2(self.down1(enc))
        up = self.up1(hidden)
        if up.size(-1) != x.size(-1):
            up = nn.functional.interpolate(up, size=x.size(-1), mode="nearest")
        return self.out_conv(self.dec1(torch.cat([up, enc], dim=1)))


class DiffusionModel(nn.Module):
    """Conditional DDPM-style model used for semantic-preserving augmentation."""

    def __init__(self, feature_dim, length_seq, timesteps, num_classes=6, embed_dim=64, device=torch.device("cpu")):
        super().__init__()
        self.feature_dim = feature_dim
        self.length_seq = length_seq
        self.timesteps = timesteps
        self.device = device
        self.num_classes = num_classes
        self.use_label = num_classes > 0
        self.betas = torch.linspace(1e-5, 1e-2, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod).to(device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod).to(device)
        self.time_embedding = nn.Embedding(timesteps, embed_dim).to(device)
        self.time_proj = nn.Linear(embed_dim, feature_dim).to(device)
        if self.use_label:
            self.label_embedding = nn.Embedding(num_classes, embed_dim).to(device)
            self.label_proj = nn.Linear(embed_dim, feature_dim).to(device)
            self.fusion_conv = nn.Conv1d(feature_dim * 3, feature_dim, 1).to(device)
        else:
            self.label_embedding = None
            self.label_proj = None
            self.fusion_conv = nn.Conv1d(feature_dim * 2, feature_dim, 1).to(device)
        self.unet = UNet1D(in_channels=feature_dim).to(device)

    def forward_diffusion(self, x, t):
        batch_size = x.shape[0]
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(batch_size, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(batch_size, 1, 1)
        epsilon = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus * epsilon, epsilon

    def forward(self, x_noisy, t, y=None):
        time_embed = self.time_proj(self.time_embedding(t)).unsqueeze(-1).expand(-1, -1, x_noisy.size(-1))
        if self.use_label and y is not None:
            label_embed = self.label_proj(self.label_embedding(y)).unsqueeze(-1).expand(-1, -1, x_noisy.size(-1))
            x_concat = torch.cat([x_noisy, time_embed, label_embed], dim=1)
        else:
            x_concat = torch.cat([x_noisy, time_embed], dim=1)
        return self.unet(self.fusion_conv(x_concat))

    @torch.no_grad()
    def reverse_diffusion_single_step(self, x, t, y=None):
        eps = self.forward(x, t, y)
        batch_size = x.shape[0]
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(batch_size, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(batch_size, 1, 1)
        return (x - sqrt_one_minus * eps) / sqrt_alpha

    @torch.no_grad()
    def ddim_reverse(self, x_t, t_start, y=None, ddim_steps=10):
        batch_size = x_t.shape[0]
        x = x_t.clone()
        t_max = int(t_start.max().item())
        steps = torch.linspace(t_max, 0, ddim_steps + 1).long().clamp(0, self.timesteps - 1)
        for step_index in range(len(steps) - 1):
            current_step = steps[step_index].item()
            previous_step = steps[step_index + 1].item()
            t_tensor = torch.full((batch_size,), current_step, device=x.device, dtype=torch.long)
            eps = self.forward(x, t_tensor, y)
            alpha_current = self.alpha_cumprod[current_step].view(1, 1, 1)
            alpha_previous = self.alpha_cumprod[previous_step].view(1, 1, 1) if previous_step > 0 else torch.ones(1, 1, 1, device=x.device)
            x0_pred = (x - (1 - alpha_current).sqrt() * eps) / alpha_current.sqrt()
            x = alpha_previous.sqrt() * x0_pred.clamp(-3, 3) + (1 - alpha_previous).sqrt() * eps
        return x

    @torch.no_grad()
    def augment_sample(self, x, y=None, t_rand=None, ddim_steps=1):
        if t_rand is None:
            t_rand = torch.randint(low=0, high=self.timesteps, size=(x.shape[0],), device=x.device)
        x_t, _ = self.forward_diffusion(x, t_rand)
        if ddim_steps <= 1:
            return self.reverse_diffusion_single_step(x_t, t_rand, y), x_t
        return self.ddim_reverse(x_t, t_rand, y, ddim_steps=ddim_steps), x_t


def min_max_normalize(data, min_vals, max_vals, epsilon=1e-8):
    return (data - min_vals) / (max_vals - min_vals + epsilon)


def load_training_data(data_path, training_mode, config):
    file_name = "train.pt" if training_mode == "diffusion_train_unlabeled" else "train_2p_labeled.pt"
    dataset = torch.load(os.path.join(data_path, file_name), map_location="cpu", weights_only=True)
    if config.use_normalization:
        samples = dataset["samples"].float()
        min_vals = samples.amin(dim=(0, 2), keepdim=True)
        max_vals = samples.amax(dim=(0, 2), keepdim=True)
        dataset["samples"] = min_max_normalize(samples, min_vals, max_vals)
    return DataLoader(MyDataset(dataset), batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=0)


def train_diffusion_model(model, dataloader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x = batch["sample"].to(device)
            y = None if not model.use_label else batch["labels"].to(device)
            t = torch.randint(low=0, high=model.timesteps, size=(x.size(0),), device=device)
            noisy_x, epsilon = model.forward_diffusion(x, t)
            loss = criterion(model(noisy_x, t, y), epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / max(1, len(dataloader)):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the DK-Root diffusion augmentation model")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--training_mode", default="diffusion_train_labeled", type=str)
    parser.add_argument("--selected_dataset", default=".", type=str)
    parser.add_argument("--data_path", default=os.path.join(os.getcwd(), "dataloader", "data_example"), type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--uncond", action="store_true", default=False)
    parser.add_argument("--diffusion_timesteps", default=None, type=int)
    parser.add_argument("--diffusion_num_epochs", default=None, type=int)
    parser.add_argument("--diffusion_lr", default=None, type=float)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = Config()
    config.device = args.device
    if args.diffusion_timesteps is not None:
        config.Diffusion.timesteps = args.diffusion_timesteps
    if args.diffusion_num_epochs is not None:
        config.Diffusion.num_epochs = args.diffusion_num_epochs
    if args.diffusion_lr is not None:
        config.Diffusion.lr = args.diffusion_lr
    data_path = os.path.normpath(os.path.join(args.data_path, args.selected_dataset))
    train_loader = load_training_data(data_path, args.training_mode, config)
    num_classes = 0 if args.uncond else config.num_classes
    save_path = config.Diffusion.save_path_diffusion_uncond_template.format(seed=args.seed) if args.uncond else config.Diffusion.save_path_diffusion_template.format(seed=args.seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = DiffusionModel(config.input_channels, config.sequence_length, config.Diffusion.timesteps, num_classes=num_classes, device=device)
    train_diffusion_model(model, train_loader, config.Diffusion.num_epochs, config.Diffusion.lr, device)
    torch.save(model.state_dict(), save_path)
    print(f"Saved diffusion checkpoint to {save_path}")


if __name__ == "__main__":
    main()
