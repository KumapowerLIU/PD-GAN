# TODO: update the testing code ASAp.
import cv2
import os
from config.test_config import TestConfig

basic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
import sys

sys.path.append(basic_dir)
import util.util as util
from models.models import create_model

if __name__ == "__main__":
    cfg = TestConfig().create_config()
    util.mkdir(cfg.results_dir)
    test_image_path = cfg.test_image
    mask_path = cfg.mask_image
    test_image_list = util.test_input_image_process(test_image_path)
    mask_list = util.test_mask_process(test_image_path)
    model = create_model(cfg)
    for iter_num in range(len(test_image_path)):
        test_image_name = test_image_list[iter_num]["img_name"]
        test_image_content = test_image_list[iter_num]["img_content"]
        # we set the 0th mask as default
        mask_content = mask_list[0]
        model.set_input(mask_content, test_image_content)
        visual_dict = model.test_forward()
        image_save_path = os.path.join(cfg.results_dir, f"{test_image_name}_test.jpg")
        util.visualize_grid(visual_dict, image_save_path, return_gird=False)
