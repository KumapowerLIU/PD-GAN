import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.SPDNorm import SPDNormResnetBlock

# SHGenerator soft and hard
class SPDNormGenerator(nn.Module):
    """
    First, transfer the random vector z with an fc layer.
    Then,
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nf = self.cfg.ngf
        self.sw = self.cfg.latent_sw
        self.sh = self.cfg.latent_sh
        # calculate the random noise size
        self.fc = nn.Linear(cfg.z_dim, 16 * nf * self.sw * self.sh)
        # progression generator
        self.generated = nn.ModuleList(
            [
                SPDNormGeneratorUnit(16 * nf, 16 * nf, 2, 3, cfg),  # in 4 out 8
                SPDNormGeneratorUnit(16 * nf, 16 * nf, 3, 3, cfg),  # in 8 out 16
                SPDNormGeneratorUnit(16 * nf, 8 * nf, 4, 3, cfg),  # in 16 out 32
                SPDNormGeneratorUnit(8 * nf, 4 * nf, 5, 3, cfg),  # in 32 out 64
                SPDNormGeneratorUnit(4 * nf, 2 * nf, 6, 5, cfg),  # in 64 out 128
                SPDNormGeneratorUnit(2 * nf, 1 * nf, 7, 5, cfg),  # in 128 out 256
            ]
        )
        self.conv_img = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, z, pre_image, mask):
        latent_v = self.fc(z)
        latent_v = latent_v.view(-1, 16 * self.cfg.ngf, self.sh, self.sw)
        input_mask = mask[:, 0, :, :].unsqueeze(1)
        out = latent_v
        for i, conv in enumerate(self.generated):
            out = conv(out, pre_image, input_mask)
        out = self.conv_img(F.leaky_relu(out, 2e-1))
        out = F.tanh(out)
        return out


class SPDNormGeneratorUnit(nn.Module):
    def __init__(self, in_channels, out_channels, mask_number, mask_ks, cfg):
        super().__init__()
        self.block = SPDNormResnetBlock(
            in_channels, out_channels, mask_number, mask_ks, cfg
        )
        self.up = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, prior_f, mask):
        out = self.block(x, prior_f, mask)
        out = self.up(out)
        return out
