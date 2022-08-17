import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np


# Ref:https://github.com/Boyiliee/Positional-Normalization
def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class SPDNormResnetBlock(nn.Module):
    def __init__(self, fin, fout, mask_number, mask_ks, cfg):
        super().__init__()
        nhidden = 128
        fmiddle = min(fin, fout)
        lable_nc = 3
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        self.learned_shortcut = True
        # apply spectral norm if specified
        if "spectral" in cfg.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPDNorm(fin, norm_type="position")
        self.norm_1 = SPDNorm(fmiddle, norm_type="position")
        self.norm_s = SPDNorm(fin, norm_type="position")
        # define the mask weight
        self.v = nn.Parameter(torch.zeros(1))
        self.activeweight = nn.Sigmoid()
        # define the feature and mask process conv
        self.mask_number = mask_number
        self.mask_ks = mask_ks
        pw_mask = int(np.ceil((self.mask_ks - 1.0) / 2))
        self.mask_conv = MaskGet(1, 1, kernel_size=self.mask_ks, padding=pw_mask)
        self.conv_to_f = nn.Sequential(
            nn.Conv2d(lable_nc, nhidden, kernel_size=3, padding=1),
            nn.InstanceNorm2d(nhidden),
            nn.ReLU(),
            nn.Conv2d(nhidden, fin, kernel_size=3, padding=1),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fin * 2, fin, kernel_size=3, padding=1), nn.Sigmoid()
        )

    # the semantic feature prior_f form pretrained encoder
    def forward(self, x, prior_image, mask):
        """

        Args:
            x: input feature
            prior_image: the output of PCConv
            mask: mask


        """
        # prepare the forward
        b, c, h, w = x.size()
        prior_image_resize = F.interpolate(
            prior_image, size=x.size()[2:], mode="nearest"
        )
        mask_resize = F.interpolate(mask, size=x.size()[2:], mode="nearest")
        # Mask Original and Res path  res weight/ value attention
        prior_feature = self.conv_to_f(prior_image_resize)
        soft_map = self.attention(torch.cat([prior_feature, x], 1))
        soft_map = (1 - mask_resize) * soft_map + mask_resize
        # Mask weight for next process
        mask_pre = mask_resize
        hard_map = 0.0
        for i in range(self.mask_number):
            mask_out = self.mask_conv(mask_pre)
            mask_generate = (mask_out - mask_pre) * (
                1 / (torch.exp(torch.tensor(i + 1).cuda()))
            )
            mask_pre = mask_out
            hard_map = hard_map + mask_generate
        hard_map_inner = (1 - mask_out) * (1 / (torch.exp(torch.tensor(i + 1).cuda())))
        hard_map = hard_map + mask_resize + hard_map_inner
        soft_out = self.conv_s(self.norm_s(x, prior_image_resize, soft_map))
        hard_out = self.conv_0(self.actvn(self.norm_0(x, prior_image_resize, hard_map)))
        hard_out = self.conv_1(
            self.actvn(self.norm_1(hard_out, prior_image_resize, hard_map))
        )
        # Res add
        out = soft_out + hard_out
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPDNorm(nn.Module):
    def __init__(self, norm_channel, norm_type="batch"):
        super().__init__()
        label_nc = 3
        param_free_norm_type = norm_type
        ks = 3
        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_channel, affine=False)
        elif param_free_norm_type == "position":
            self.param_free_norm = PositionalNorm2d
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        pw = ks // 2
        nhidden = 128
        self.mlp_activate = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)

    def forward(self, x, prior_f, weight):
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on condition feature
        actv = self.mlp_activate(prior_f)
        gamma = self.mlp_gamma(actv) * weight
        beta = self.mlp_beta(actv) * weight
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class MaskGet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

    def forward(self, input):
        # depart from partial conv
        # hole region should sed to 0
        with torch.no_grad():
            output_mask = self.mask_conv(input)
        no_update_holes = output_mask == 0
        new_mask = torch.ones_like(output_mask)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        return new_mask


def get_nonspade_norm_layer(cfg, norm_type="instance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith("spectral"):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError("normalization layer %s is not recognized" % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer
