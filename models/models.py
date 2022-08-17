def create_model(cfg):
    """
    Choose the type of model. There two types of model: 'PCConv' and 'PD-GAN'. The first one is the model that
    we used in the first stage in our paper.
    """
    if cfg.model == "PConv":
        from .PConv import PCConv

        model = PCConv(cfg)
    if cfg.model == "PDGAN":
        from .PDGAN import PDGAN

        model = PDGAN(cfg)
    print("model [%s] was created" % (model.name()))
    return model
