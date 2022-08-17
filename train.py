import os
import numpy as np
import time
from datetime import datetime
from config.train_config import TrainConfig
from data.dataprocess import Dataset
from models.models import create_model
from util.util import visualize_grid, mkdir
from torch.utils import data
from loguru import logger

basic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
import sys

sys.path.append(basic_dir)
if __name__ == "__main__":
    cfg = TrainConfig().create_config()
    result_save_base_dir = os.path.join(basic_dir, cfg.checkpoints_dir, cfg.name)
    logger.add(os.path.join(result_save_base_dir, cfg.log_dir, "train.log"))
    visual_save_base = os.path.join(result_save_base_dir, "visuals")
    mkdir(visual_save_base)
    if cfg.write_summary:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(result_save_base_dir, cfg.log_dir))
    logger.info("Define the dataset")
    dataset = Dataset(cfg.gt_root, cfg, mask_file=cfg.mask_root)
    iterator_train = data.DataLoader(
        dataset, batch_size=cfg.batchSize, shuffle=True, num_workers=cfg.num_workers
    )
    logger.info("Create model")
    model = create_model(cfg)
    total_steps = 0
    logger.info("Start training")
    for epoch in range(cfg.epoch_count, cfg.niter + cfg.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for gt, mask in iterator_train:
            iter_start_time = time.time()
            total_steps += cfg.batchSize
            epoch_iter += cfg.batchSize
            model.set_input(mask, gt)
            model.optimize_parameters()
            # display the training processing
            if total_steps % cfg.display_freq == 0:
                visual_dict = model.get_current_visuals()
                image_save_path = os.path.join(
                    visual_save_base, f"{total_steps:06}.jpg"
                )
                grid_image = visualize_grid(
                    visual_dict, image_save_path, return_gird=True
                )
                writer.add_image(
                    "train_images",
                    (grid_image / 255.0).astype(np.float32).transpose(2, 0, 1),
                    total_steps,
                )
            # display the training loss
            if total_steps % cfg.print_freq == 0:
                losses = model.get_current_errors()
                loss_info = f"ExpName: {cfg.name} \nEpoch: {epoch}, Steps: {total_steps}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                for k, v in losses.items():
                    loss_info = loss_info + f"{k}: {v.item():.4f}, "
                    if cfg.write_summary:
                        writer.add_scalar("train_loss/" + k, v, global_step=total_steps)
                logger.info(loss_info)
        if epoch % cfg.save_epoch_freq == 0:
            save_info = "saving the model at the end of epoch {}, iters {}".format(
                epoch, total_steps
            )
            logger.info(save_info)
            model.save_networks(epoch)
        logger.info(
            "End of epoch {} / {} \t Time Taken: {} sec".format(
                epoch, cfg.niter + cfg.niter_decay, time.time() - epoch_start_time
            )
        )
        model.update_learning_rate(epoch)
