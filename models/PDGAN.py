import torch
from collections import OrderedDict
from .base_model import BaseModel
from .network import networks
from models.blocks.loss import GANLoss, Diversityloss, PerceptualLoss, TVloss, StyleLoss
from .testmaskconv import MaskEncoder
from torchvision import utils
from loguru import logger
from functools import reduce


class PDGAN(BaseModel):
    def __init__(self, cfg):
        super(PDGAN, self).__init__(cfg)
        self.isTrain = cfg.isTrain
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.netPDGAN, self.netEN, self.netDE = networks.define_G(
            cfg, self.select_model, cfg.norm, cfg.init_type, self.gpu_ids, cfg.init_gain
        )

        self.model_names = ["PDGAN"]
        logger.info("network {} has been defined".format(self.select_model))
        self.set_requires_grad(self.netEN, False)
        self.set_requires_grad(self.netDE, False)
        try:
            ckpt_PConvEN = torch.load(cfg.pre_ckpt_PConv_EN_dir)
            ckpt_PConvDE = torch.load(cfg.pre_ckpt_PConv_DE_dir)
            self.netEN.module.load_state_dict(ckpt_PConvEN["net"])
            self.netDE.module.load_state_dict(ckpt_PConvDE["net"])
            logger.info(
                "network {} has been defined and the pretrain model {} has been load".format(
                    "PConvEN", cfg.pre_ckpt_PConv_EN_dir
                )
            )
            logger.info(
                "network {} has been defined and the pretrain model {} has been load".format(
                    "PConvDE", cfg.pre_ckpt_PConv_DE_dir
                )
            )
        except:
            logger.info(
                "We can not define the pretrained PartialConv, please check the save path of pretrained model"
            )
            raise ValueError("Wrong path of pretrained PartialConv")
        if self.isTrain:
            self.old_lr = cfg.lr
            # define loss
            # self.PerceptualLoss = PerceptualLoss(weights=[1.0, 1.0, 1.0, 1.0, 1.0])
            self.PerceptualLoss = PerceptualLoss(weights=[1.0, 1.0, 1.0, 1.0, 0.0])
            self.Diversityloss = Diversityloss()
            self.StyleLoss = StyleLoss()
            # self.Diversityloss = DiversityStyleLoss()
            self.criterionGAN = GANLoss(cfg.gan_mode, tensor=self.Tensor, cfg=self.cfg)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.schedulers = []
            self.optimizers = []
            # define discriminator
            self.netD = networks.define_D(
                self.cfg, cfg.init_type, self.gpu_ids, cfg.init_gain
            )
            self.model_names.append("D")
            logger.info(
                "network {} has been defined".format("Multi Scale Discriminator")
            )
            self.optimizer_PDGAN = torch.optim.Adam(
                self.netPDGAN.parameters(), lr=cfg.lr / 2, betas=(cfg.beta1, 0.9)
            )
            self.optimizer_D = torch.optim.Adam(
                list(self.netD.parameters()), lr=cfg.lr * 2, betas=(cfg.beta1, 0.9)
            )
            self.optimizers.append(self.optimizer_PDGAN)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, cfg))
        logger.info("---------- Networks initialized -------------")
        networks.print_network(self.netPDGAN)
        if self.isTrain:
            networks.print_network(self.netD)
        logger.info("---------- Networks initialized Done-------------")
        if not self.isTrain or cfg.continue_train:
            logger.info(
                "Loading pre-trained network {}! You choose the results of {} epoch".format(
                    self.select_model, cfg.which_epoch
                )
            )
            self.load_networks(cfg.which_epoch)

    def name(self):
        return self.select_model

    def set_input(self, mask, gt):
        """
        Args:
            mask: input mask, the pixel value of masked regions is 1
            gt: ground truth image

        """
        self.gt = gt.to(self.device)
        self.mask = mask.to(self.device)
        self.input_img = gt.clone()
        if mask.shape[1] == 1:
            self.mask = self.mask.repeat(1, 3, 1, 1)
        self.inv_mask = 1 - self.mask
        self.input_img = self.gt.clone()
        self.input_img = self.input_img * self.inv_mask

    def forward(self):
        """

        There two outputs at the same time with two different z, respectively. You can find the purpose in the
        sec.3.2 of paper.


        """
        b, c, h, w = self.gt.size()
        fake_p_6, fake_p_5, fake_p_4, fake_p_3, fake_p_2, fake_p_1 = self.netEN(
            [self.input_img, self.inv_mask]
        )
        self.pre_image = self.netDE(
            fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6
        )
        # self.pre_image = self.netPCConv([self.input_img, self.inv_mask])
        self.pre_image = self.pre_image.detach()
        self.z_A = torch.randn(
            b, self.cfg.z_dim, dtype=torch.float32, device=self.gt.get_device()
        )
        self.z_B = torch.randn(
            b, self.cfg.z_dim, dtype=torch.float32, device=self.gt.get_device()
        )

        self.G_out = self.netPDGAN(
            torch.cat((self.z_A, self.z_B), 0),
            torch.cat((self.pre_image, self.pre_image), 0),
            torch.cat((self.inv_mask, self.inv_mask), 0),
        )
        self.fake_A, self.fake_B = torch.split(self.G_out, b, dim=0)

    def backward_D(self):
        real = self.gt
        b, c, h, w = self.gt.size()
        fake_A, fake_B = torch.split(self.G_out.detach(), b, dim=0)
        pred_fake_A, pred_real_A = self.discriminate(self.mask.float(), fake_A, real)
        pred_fake_B, pred_real_B = self.discriminate(self.mask.float(), fake_B, real)

        D_losses_fake = reduce(
            lambda x, y: x + y,
            [
                self.criterionGAN(pred_fake_A, False, for_discriminator=True),
                self.criterionGAN(pred_fake_B, False, for_discriminator=True),
            ],
        )
        D_losses_real = reduce(
            lambda x, y: x + y,
            [
                self.criterionGAN(pred_real_A, True, for_discriminator=True),
                self.criterionGAN(pred_real_B, True, for_discriminator=True),
            ],
        )
        self.lossD = D_losses_fake + D_losses_real
        self.lossD.backward(retain_graph=True)

    def backward_G(self):

        G_losses = {}

        # First, The generator should fake the discriminator
        real = self.gt
        b, c, h, w = self.gt.size()
        fake_A, fake_B = torch.split(self.G_out, b, dim=0)
        pred_fake_A, pred_real_A = self.discriminate(self.mask.float(), fake_A, real)
        pred_fake_B, pred_real_B = self.discriminate(self.mask.float(), fake_B, real)
        G_losses["GAN"] = reduce(
            lambda x, y: x + y,
            [
                self.criterionGAN(pred_fake_A, True, for_discriminator=False),
                self.criterionGAN(pred_fake_B, True, for_discriminator=False),
            ],
        )
        # feature matching loss, ref:https://github.com/NVIDIA/pix2pixHD
        if not self.cfg.no_ganFeat_loss:
            num_D = len(pred_fake_A)
            GAN_Feat_loss = self.Tensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake_A[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake_A[i][j], pred_real_A[i][j].detach()
                    ) + self.criterionFeat(
                        pred_fake_B[i][j], pred_real_B[i][j].detach()
                    )
                    GAN_Feat_loss += unweighted_loss * self.cfg.lambda_feat / num_D
            G_losses["GAN_Feat"] = GAN_Feat_loss
        Completion_A = self.fake_A * self.mask + self.inv_mask * self.gt
        Completion_B = self.fake_B * self.mask + self.inv_mask * self.gt
        G_losses["TV"] = (
            reduce(
                lambda x, y: x + y,
                [
                    TVloss(Completion_A, self.inv_mask, "mean"),
                    TVloss(Completion_B, self.inv_mask, "mean"),
                ],
            )
            * 0.1
        )

        G_losses["VGG"] = (
            reduce(
                lambda x, y: x + y,
                [
                    self.PerceptualLoss(fake_A, self.gt),
                    self.PerceptualLoss(fake_B, self.gt),
                ],
            )
            * self.cfg.lambda_P
        )
        G_losses["Diversity"] = 1 / (
            self.Diversityloss(fake_A * self.mask, fake_B * self.mask) + (1 * 1e-5)
        )
        loss_G = sum(G_losses.values()).mean()
        self.lossG = G_losses
        loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # optimize the D first
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netPDGAN, False)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize Generate net
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netPDGAN, True)
        self.optimizer_PDGAN.zero_grad()
        self.backward_G()
        self.optimizer_PDGAN.step()

    def get_current_errors(self):
        # show the current loss
        return OrderedDict(
            [
                ("Perceptual", self.lossG["VGG"]),
                ("GAN_Feat", self.lossG["GAN_Feat"]),
                ("Diversity", self.lossG["Diversity"]),
                ("Generator", self.lossG["GAN"]),
                ("Discriminator", self.lossD),
            ]
        )

    def get_current_visuals(self):

        return {
            "input_image": self.input_img,
            "mask": self.mask,
            "ground_truth": self.gt,
            "pconv_out": self.pre_image,
            "fake_A": self.fake_A,
            "fake_B": self.fake_B,
        }

    def discriminate(self, input_mask, fake_image, real_image):
        fake_concat = torch.cat([input_mask, fake_image], dim=1)
        real_concat = torch.cat([input_mask, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2 :]

        return fake, real

    def test_forward(self, image_num=2):
        """

        There two outputs at the same time with two different z, respectively. You can find the purpose in the
        sec.3.2 of paper.


        """
        out_dict = {}
        with torch.no_grad():
            b, c, h, w = self.gt.size()
            fake_p_6, fake_p_5, fake_p_4, fake_p_3, fake_p_2, fake_p_1 = self.netEN(
                [self.input_img, self.inv_mask]
            )
            self.pre_image = self.netDE(
                fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6
            ).detach()
            for i in range(image_num):
                z = torch.randn(
                    b, self.cfg.z_dim, dtype=torch.float32, device=self.gt.get_device()
                )
                out = self.netPDGAN(z, self.pre_image, self.inv_mask)
                out_dict[f"out_{i}"] = out
        out_dict["input_image"] = self.input_img
        out_dict["mask"] = self.mask
        out_dict["ground_truth"] = self.gt
        out_dict["pconv_out"] = self.pre_image
        return out_dict
