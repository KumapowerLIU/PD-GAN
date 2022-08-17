import torch.nn as nn
import torch


class PartialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        input = x[0]
        mask = x[1].float().cuda()

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = [output, new_mask]
        return out


class PCBActiv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        norm_layer="instance",
        sample="down-4",
        activ="leaky",
        conv_bias=False,
        inner=False,
        outer=False,
    ):
        super().__init__()
        if sample == "same-5":
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == "same-7":
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == "down-4":
            self.conv = PartialConv(in_ch, out_ch, 4, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if norm_layer == "instance":
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm_layer == "batch":
            self.norm = nn.BatchNorm2d(out_ch, affine=True)
        else:
            pass

        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            pass
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.activation(out[0])
            out = self.conv(out)
        elif self.outer:
            out = self.conv(out)
        else:
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.norm(out[0])
        return out


# Define the resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm="instance"):
        super(ResnetBlock, self).__init__()
        self.conv_1 = PartialConv(dim, dim, 3, 1, 1, 1)
        if norm == "instance":
            self.norm_1 = nn.InstanceNorm2d(dim, track_running_stats=False)
            self.norm_2 = nn.InstanceNorm2d(dim, track_running_stats=False)
        elif norm == "batch":
            self.norm_1 = nn.BatchNorm2d(dim, track_running_stats=False)
            self.norm_2 = nn.BatchNorm2d(dim, track_running_stats=False)
        self.active = nn.ReLU(True)
        self.conv_2 = PartialConv(dim, dim, 3, 1, 1, 1)

    def forward(self, x):
        out = self.conv_1(x)
        out[0] = self.norm_1(out[0])
        out[0] = self.active(out[0])
        out = self.conv_2(out)
        out[0] = self.norm_2(out[0])
        out[0] = x[0] + out[0]
        return out


class UnetSkipConnectionDBlock(nn.Module):
    def __init__(
        self,
        inner_nc,
        outer_nc,
        outermost=False,
        innermost=False,
        norm_layer="instance",
    ):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU()
        upconv = nn.ConvTranspose2d(
            inner_nc, outer_nc, kernel_size=4, stride=2, padding=1
        )
        if norm_layer == "instance":
            upnorm = nn.InstanceNorm2d(outer_nc, affine=True)
        elif norm_layer == "batch":
            upnorm = nn.BatchNorm2d(outer_nc, affine=True)
        else:
            pass

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
