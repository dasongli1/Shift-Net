# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F
import math
import os

from basicsr.data.transforms import mod_crop
from basicsr.utils import img2tensor, scandir
import random

def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img

    return [_augment(a) for a in args]


def postprocess(*images, rgb_range, ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        if ycbcr_flag:
            out = img.mul(rgb_coefficient).clamp(16, 235)
        else:
            out = img.mul(rgb_coefficient).clamp(0, 255).round()

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]


def calc_psnr(img1, img2, rgb_range=1., shave=4):
    if isinstance(img1, torch.Tensor):
        img1 = img1[:, :, shave:-shave, shave:-shave]
        img1 = img1.to('cpu').numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2[:, :, shave:-shave, shave:-shave]
        img2 = img2.to('cpu').numpy()
    mse = np.mean((img1 / rgb_range - img2 / rgb_range) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_grad_sobel(img, device='cuda'):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = torch.mean(img, dim=1, keepdim=True)

    # img = calc_meanFilter(img, device=device)  # meanFilter

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = torch.from_numpy(sobel_filter_X).float().to(device)
    sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().to(device)
    grad_X = F.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_meanFilter(img, kernel_size=11, n_channel=1, device='cuda'):
    mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    mean_filter_X = torch.from_numpy(mean_filter_X).float().to(device)
    new_img = torch.zeros_like(img)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                             stride=1, padding=kernel_size // 2)
    return new_img

def read_img_seq(path, require_mod_crop=False, scale=1):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    return imgs


def generate_frame_indices(crt_idx,
                           max_frame_num,
                           num_frames,
                           padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle',
                       'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
        0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
