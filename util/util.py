from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import random
import torchvision
import inspect, re
import numpy as np
import os
import collections
import cv2
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from natsort import natsorted


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def binary_mask(in_mask, threshold):
    assert in_mask.dim() == 2, "mask must be 2 dimensions"

    output = torch.ByteTensor(in_mask.size())
    output = (output > threshold).float().mul_(1)

    return output


def cal_feat_mask(inMask, nlayers):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    inMask = inMask.float()
    ntimes = 2**nlayers
    inMask = F.interpolate(
        inMask, (inMask.size(2) // ntimes, inMask.size(3) // ntimes), mode="nearest"
    )
    inMask = inMask.detach().byte()

    return inMask


# sp_x: LongTensor
# sp_y: LongTensor
def cal_sps_for_Advanced_Indexing(h, w):
    sp_y = torch.arange(0, w).long()
    sp_y = torch.cat([sp_y] * h)

    lst = []
    for i in range(h):
        lst.extend([i] * w)
    sp_x = torch.from_numpy(np.array(lst))
    return sp_x, sp_y


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [
        e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)
    ]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print(
        "\n".join(
            [
                "%s %s"
                % (
                    method.ljust(spacing),
                    processFunc(str(getattr(object, method).__doc__)),
                )
                for method in methodList
            ]
        )
    )


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r"\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_grid(visdict, savepath=None, size=256, dim=1, return_gird=True):
    """
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    """
    assert dim == 1 or dim == 2
    grids = {}
    for key in visdict:
        _, _, h, w = visdict[key].shape
        if dim == 2:
            new_h = size
            new_w = int(w * size / h)
        elif dim == 1:
            new_h = int(h * size / w)
            new_w = size
        grids[key] = torchvision.utils.make_grid(
            F.interpolate(visdict[key], [new_h, new_w]).detach().cpu()
        )
    grid = (torch.cat(list(grids.values()), 1) + 1) / 2
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
    if savepath:
        cv2.imwrite(savepath, grid_image)
    if return_gird:
        return grid_image


def get_file(filepath, state="rb"):
    """
    Args:
        filepath: the input path of file
        state: the mode of data reading

    Returns:
        the content of file with the reading mode
    """
    filepath = str(filepath)
    with open(filepath, state) as f:
        value_buf = f.read()
    return value_buf


def imread_bytes(content, flag="unchanged"):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if img.shape[1] != 512:
        print("We need resize")
        img = cv2.resize(img, (512, 512))

    return img


def input_process(img_path, image_size, device=torch.device("cuda"), mask=False):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = (
        (image / 255.0)
        .reshape(image_size * image_size, 3)
        .transpose()
        .reshape(3, image_size, image_size)
    )
    torch_image = torch.from_numpy(image).float().to(device)
    if mask:
        return (torch_image > 0).float()
    else:
        return torch_image * 2.0 - 1.0


def test_mask_process(mask_path):
    mask_list = []
    if mask_path.endswith((".png", ".jpg", ".jpeg", "JPG", "bmp")):
        mask_list.append(input_process(mask_path, image_size=256, mask=True))

    elif len(os.listdir(mask_path)) > 0:
        img_name_list = natsorted(
            [
                img_name
                for img_name in os.listdir(mask_path)
                if img_name.endswith((".png", ".jpg", ".jpeg", ".JPG", ".bmp"))
            ]
        )

        img_nums = len(img_name_list)
        if img_nums == 0:
            raise ValueError(f"The {mask_path} has no image, plead check your folder")
        else:
            for _, img_path_each in enumerate(img_name_list):
                mask_list.append(input_process(mask_path, image_size=256, mask=True))
    return mask_list


def test_input_image_process(img_path):
    img_list = []
    if img_path.endswith((".png", ".jpg", ".jpeg", "JPG", "bmp")):

        img_pair = {
            "img_name": os.path.basename(img_path).split(".")[0],
            "img_content": input_process(img_path, image_size=256),
        }
        img_list.append(img_pair)

    elif len(os.listdir(img_path)) > 0:
        img_name_list = natsorted(
            [
                img_name
                for img_name in os.listdir(img_path)
                if img_name.endswith((".png", ".jpg", ".jpeg", ".JPG", ".bmp"))
            ]
        )

        img_nums = len(img_name_list)
        if img_nums == 0:
            raise ValueError(f"The {img_path} has no image, plead check your folder")
        else:
            for _, img_path_each in enumerate(img_name_list):
                img_pair = {
                    "img_name": os.path.basename(img_path_each).split(".")[0],
                    "img_content": input_process(img_path_each, image_size=256),
                }
                img_list.append(img_pair)

    return img_list
