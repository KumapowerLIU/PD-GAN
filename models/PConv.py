import torch
from collections import OrderedDict
from .base_model import BaseModel
from .network import networks
from models.blocks.loss import PerceptualLoss, StyleLoss, TVloss
from loguru import logger


class PCConv(BaseModel):
    def __init__(self, cfg):
        super(PCConv, self).__init__(cfg)
        self.isTrain = cfg.isTrain
        self.cfg = cfg
        # define network
        self.netEN, self.netDE = networks.define_G(
            cfg, self.select_model, cfg.norm, cfg.init_type, self.gpu_ids, cfg.init_gain
        )
        self.model_names = ["EN", "DE"]
        logger.info("network {} has been defined".format(self.select_model))
        if self.isTrain:
            self.old_lr = cfg.lr
            # define loss functions
            self.PerceptualLoss = PerceptualLoss()
            self.StyleLoss = StyleLoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_EN = torch.optim.Adam(
                self.netEN.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999)
            )
            self.optimizer_DE = torch.optim.Adam(
                self.netDE.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_EN)
            self.optimizers.append(self.optimizer_DE)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, cfg))

        logger.info("---------- Networks initialized -------------")
        networks.print_network(self.netEN)
        networks.print_network(self.netDE)
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
        self.input_img = self.gt.clone()
        self.mask = self.mask.repeat(1, 3, 1, 1)
        #  unpositve with original mask
        self.inv_mask = 1 - self.mask
        # Do not set the mask regions as 0, this process can stable training
        self.input_img.narrow(1, 0, 1).masked_fill_(
            self.mask.narrow(1, 0, 1).bool(), 2 * 123.0 / 255.0 - 1.0
        )
        self.input_img.narrow(1, 1, 1).masked_fill_(
            self.mask.narrow(1, 0, 1).bool(), 2 * 104.0 / 255.0 - 1.0
        )
        self.input_img.narrow(1, 2, 1).masked_fill_(
            self.mask.narrow(1, 0, 1).bool(), 2 * 117.0 / 255.0 - 1.0
        )

    def forward(self):
        fake_p_6, fake_p_5, fake_p_4, fake_p_3, fake_p_2, fake_p_1 = self.netEN(
            [self.input_img, self.inv_mask]
        )
        self.G_out = self.netDE(
            fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6
        )
        self.Completion = self.G_out * self.mask + self.inv_mask * self.gt

    def backward_G(self):
        self.hole_loss = self.criterionL1(self.G_out * self.mask, self.gt * self.mask)
        self.valid_loss = self.criterionL1(
            self.G_out * self.inv_mask, self.gt * self.inv_mask
        )
        self.Perceptual_loss = self.PerceptualLoss(
            self.G_out, self.gt
        ) + self.PerceptualLoss(self.Completion, self.gt)
        self.Style_Loss = self.StyleLoss(self.G_out, self.gt)
        self.TV_Loss = TVloss(self.Completion, self.inv_mask, "mean")
        # The weights of losses are same as
        # https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf
        self.loss_G = (
            self.hole_loss * 6
            + self.Perceptual_loss * 0.05
            + self.Style_Loss * 120
            + self.TV_Loss * 0.1
            + self.valid_loss
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_EN.zero_grad()
        self.optimizer_DE.zero_grad()
        self.backward_G()
        self.optimizer_EN.step()
        self.optimizer_DE.step()

    def get_current_errors(self):
        # show the current loss
        return OrderedDict(
            [
                ("L1_valid", self.valid_loss),
                ("L1_hole", self.hole_loss),
                ("Style", self.Style_Loss),
                ("Perceptual", self.Perceptual_loss),
                ("TV_Loss", self.TV_Loss),
            ]
        )

    # You can also see the Tensorborad
    def get_current_visuals(self):
        return {
            "input_image": self.input_img,
            "output": self.Completion,
            "mask": self.mask,
            "ground_truth": self.gt,
        }
