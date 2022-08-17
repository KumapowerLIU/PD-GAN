# Define networks, init networks
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .pconv import Encoder, Decoder
from models.network.Discriminator import MultiscaleDiscriminator
from .pdgan import SPDNormGenerator


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, cfg):
    if cfg.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + cfg.epoch_count - cfg.niter) / float(
                cfg.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_decay_iters, gamma=0.1
        )
    elif cfg.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif cfg.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.niter, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", cfg.lr_policy
        )
    return scheduler


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(
    cfg, model_name, norm="batch", init_type="normal", gpu_ids=[], init_gain=0.02
):
    if model_name is "PDGAN":
        PConvEncoder = Encoder(cfg.input_nc, cfg.ngf, norm_layer=norm)
        PConvDecoder = Decoder(cfg.output_nc, cfg.ngf, norm_layer=norm)
        PDGANNet = SPDNormGenerator(cfg)
        return (
            init_net(PDGANNet, init_type, init_gain, gpu_ids),
            init_net(PConvEncoder, init_type, init_gain, gpu_ids),
            init_net(PConvDecoder, init_type, init_gain, gpu_ids),
        )
    elif model_name is "PConv":
        PConvEncoder = Encoder(cfg.input_nc, cfg.ngf, norm_layer=norm)
        PConvDecoder = Decoder(cfg.output_nc, cfg.ngf, norm_layer=norm)
        return init_net(PConvEncoder, init_type, init_gain, gpu_ids), init_net(
            PConvDecoder, init_type, init_gain, gpu_ids
        )
    else:
        raise ValueError("select wrong model name:{}".format(model_name))


def define_D(cfg, init_type="normal", gpu_ids=[], init_gain=0.02):
    netD = MultiscaleDiscriminator(cfg)
    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)
