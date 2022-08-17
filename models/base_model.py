import os
import torch


class BaseModel:
    def __init__(self, cfg):
        self.model_names = None
        self.cfg = cfg
        self.gpu_ids = cfg.gpu_ids
        self.isTrain = cfg.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )
        self.save_dir = os.path.join(cfg.basic_dir, cfg.checkpoints_dir, cfg.name)
        self.select_model = cfg.model
        self.input_img = None
        self.gt = None
        self.mask = None
        self.inv_mask = None

    def name(self):
        return "BaseModel"

    def set_input(self, **kwargs):
        pass

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename).replace(
                    "\\", "/"
                )
                net = getattr(self, "net" + name)
                optimize = getattr(self, "optimizer_" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(
                        {
                            "net": net.module.cpu().state_dict(),
                            "optimize": optimize.state_dict(),
                        },
                        save_path,
                    )
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)

                net = getattr(self, "net" + name)
                optimize = getattr(self, "optimizer_" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(
                    load_path.replace("\\", "/"), map_location=str(self.device)
                )
                optimize.load_state_dict(state_dict["optimize"])
                net.load_state_dict(state_dict["net"])

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self, epoch):
        if "PDGAN" in self.model_names:
            # We use TTUR
            if epoch > self.cfg.niter:
                lrd = self.cfg.lr / self.cfg.niter_decay
                new_lr = self.old_lr - lrd
            else:
                new_lr = self.old_lr

            if new_lr != self.old_lr:

                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

                for param_group in self.optimizer_D.param_groups:
                    param_group["lr"] = new_lr_D
                for param_group in self.optimizer_PDGAN.param_groups:
                    param_group["lr"] = new_lr_G
                print("update learning rate: %f -> %f" % (self.old_lr, new_lr))
                self.old_lr = new_lr
        elif "EN" in self.model_names or "DE" in self.model_names:
            for scheduler in self.schedulers:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]["lr"]
            print("learning rate = %.7f" % lr)
        else:
            raise ValueError(f"wrong model name, please select one of (PDGAN|PConv)")
