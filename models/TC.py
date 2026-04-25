import numpy as np
import torch
import torch.nn as nn

from .attention import Seq_Transformer


class TC(nn.Module):
    """Temporal contrast module used during contrastive pretraining."""

    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for _ in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.device = device
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )
        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim,
                                               depth=4, heads=4, mlp_dim=64)

    def forward(self, z_aug1, z_aug2):
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)
        z_aug2 = z_aug2.transpose(1, 2)
        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)
        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for step in np.arange(1, self.timestep + 1):
            encode_samples[step - 1] = z_aug2[:, t_samples + step, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]
        c_t = self.seq_transformer(forward_seq)
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for step in np.arange(0, self.timestep):
            pred[step] = self.Wk[step](c_t)
        for step in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[step], torch.transpose(pred[step], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1.0 * batch * self.timestep
        return nce, self.projection_head(c_t)
