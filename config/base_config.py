import argparse
import os
from util import util
import torch

basic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BaseConfig:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--basic_dir", type=str, default=basic_dir)
        parser.add_argument(
            "--gan_mode", type=str, default="hinge", help="(ls|original|hinge)"
        )
        parser.add_argument(
            "--gt_root",
            type=str,
            default="/workspace/project-nas-10583/cvpr2020/data/valley",
            help="path to detail images (which are the groundtruth)",
        )
        parser.add_argument(
            "--mask_root",
            type=str,
            default="/workspace/project-nas-10583/cvpr2020/data/testing_mask_dataset",
            help="path to mask, we use the datasetsets of partial conv hear",
        )
        parser.add_argument("--mask_type", type=str, default="from_file")
        parser.add_argument("--batchSize", type=int, default=8, help="input batch size")
        parser.add_argument(
            "--num_workers", type=int, default=4, help="numbers of the core of CPU"
        )
        parser.add_argument(
            "--name",
            type=str,
            default="PDGAN-Training",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument(
            "--train_image_size",
            type=int,
            default=256,
            help="image size of training process",
        )
        parser.add_argument(
            "--input_nc", type=int, default=3, help="# of input image channels"
        )
        parser.add_argument(
            "--output_nc", type=int, default=3, help="# of output image channels"
        )
        parser.add_argument(
            "--input_nc_D",
            type=int,
            default=6,
            help="# of input image channels of discriminator",
        )
        parser.add_argument(
            "--ngf", type=int, default=64, help="# of gen filters in first conv layer"
        )
        parser.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="# of discrim filters in first conv layer",
        )
        parser.add_argument(
            "--n_layers_D",
            type=int,
            default=4,
            help="only used if which_model_netD==n_layers",
        )
        parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="PDGAN",
            help="select the type of model PConv or PDGAN",
        )
        parser.add_argument(
            "--nThreads", default=2, type=int, help="# threads for loading data"
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="checkpoints",
            help="models and logs are saved here",
        )
        parser.add_argument(
            "--norm",
            type=str,
            default="instance",
            help="instance normalization or batch normalization",
        )
        parser.add_argument(
            "--use_dropout", action="store_true", help="use dropout for the generator"
        )
        parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="network initialization [normal|xavier|kaiming|orthogonal]",
        )

        parser.add_argument(
            "--lambda_L1", type=int, default=1, help="weight on L1 term in objective"
        )
        parser.add_argument(
            "--lambda_S", type=int, default=10, help="weight on Style loss in objective"
        )
        parser.add_argument(
            "--lambda_P",
            type=int,
            default=10,
            help="weight on Perceptual loss in objective",
        )
        parser.add_argument(
            "--lambda_Gan", type=int, default=1, help="weight on GAN term in objective"
        )
        parser.add_argument(
            "--lambda_TV", type=int, default=0.05, help="weight on TV loss in objective"
        )
        parser.add_argument(
            "--lambda_feat",
            type=float,
            default=10.0,
            help="weight for feature matching loss",
        )
        parser.add_argument(
            "--init_gain",
            type=float,
            default=0.02,
            help="scaling factor for normal, xavier and orthogonal.",
        )
        parser.add_argument(
            "--latent_sw", type=int, default=4, help="latent feature size"
        )
        parser.add_argument(
            "--latent_sh", type=int, default=4, help="latent feature size"
        )
        parser.add_argument(
            "--z_dim", type=int, default=256, help="dimension of the latent z vector"
        )
        parser.add_argument(
            "--norm_G",
            type=str,
            default="spectral",
            help="dimension of the latent z vector",
        )
        parser.add_argument(
            "--pre_ckpt_PConv_EN_dir", default=" ", type=str, help="max image size"
        )
        parser.add_argument(
            "--pre_ckpt_PConv_DE_dir", default=" ", type=str, help="max image size"
        )
        parser.add_argument("--num_D", default=2, type=int, help="D num")
        parser.add_argument(
            "--no_ganFeat_loss",
            action="store_true",
            help="if specified, do *not* use discriminator feature matching loss",
        )
        parser.add_argument(
            "--norm_D",
            type=str,
            default="spectralinstance",
            help="instance normalization or batch normalization",
        )

        # data_process
        parser.add_argument(
            "--need_crop", action="store_true", help="if true, cropping the images"
        )
        parser.add_argument(
            "--need_flip", action="store_true", help="if true, flipping the images"
        )
        self.initialized = True
        return parser

    def gather_config(self):
        # initialize parser with basic cfgions
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_config(self, cfg):
        message = ""
        message += "----------------- Config ---------------\n"
        for k, v in sorted(vars(cfg).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(basic_dir, cfg.checkpoints_dir, cfg.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "cfg.txt")
        with open(file_name, "wt") as cfg_file:
            cfg_file.write(message)
            cfg_file.write("\n")

    def create_config(self):

        cfg = self.gather_config()
        cfg.isTrain = self.isTrain  # train or test

        # process cfg.suffix

        self.print_config(cfg)

        # set gpu ids
        str_ids = cfg.gpu_ids.split(",")
        cfg.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                cfg.gpu_ids.append(id)
        if len(cfg.gpu_ids) > 0:
            torch.cuda.set_device(cfg.gpu_ids[0])

        self.cfg = cfg
        return self.cfg
