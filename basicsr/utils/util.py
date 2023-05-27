import os
import sys
import time
import math
import torch.nn.functional as F
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
# from utils.Demosaicing_malvar2004 import demosaicing_CFA_Bayer_Malvar2004
import pdb
import os.path as osp
from basicsr.torch_similarity.modules import NormalizedCrossCorrelation
import torchvision
# from models.archs.arch_util import flow_warp
import matplotlib.pyplot as plt

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
    
def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################

def discard_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # pdb.set_trace()
        if k[0:6] == 'module':
            name = k[7:]
            new_state_dict[name] = v
    return new_state_dict

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    # pdb.set_trace()
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################
def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def rggb_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 3D(4,H,W), RGGB channel
    Output: 3D(H,W,4), np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGGB

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def rggb2bgr(tensor, pattern='RGGB'):
    mosaic_img = np.zeros((int(tensor.shape[0]*2), int(tensor.shape[1]*2)), dtype=tensor.dtype)
    mosaic_img[0::2, 0::2] = tensor[:,:,0]
    mosaic_img[0::2, 1::2] = tensor[:,:,1]
    mosaic_img[1::2, 0::2] = tensor[:,:,2]
    mosaic_img[1::2, 1::2] = tensor[:,:,3]

    results = demosaicing_CFA_Bayer_Malvar2004(mosaic_img, pattern)
    results = np.clip(results, 0, 1)
    results = results[:, :, [2, 1, 0]]
    return results

def rggb2bayer(tensor, pattern='RGGB'):
    mosaic_img = np.zeros((int(tensor.shape[0]*2), int(tensor.shape[1]*2)), dtype=tensor.dtype)
    mosaic_img[0::2, 0::2] = tensor[:,:,0]
    mosaic_img[0::2, 1::2] = tensor[:,:,1]
    mosaic_img[1::2, 0::2] = tensor[:,:,2]
    mosaic_img[1::2, 1::2] = tensor[:,:,3]

    return mosaic_img

def bayer2bgr(tensor, pattern='RGGB'):
    results = demosaicing_CFA_Bayer_Malvar2004(tensor, pattern)
    results = np.clip(results, 0, 1)
    results = results[:, :, [2, 1, 0]]
    return results

def rgb2yuv(rgb):
    width, height, _ = rgb.shape
    yuv2rgb_matrix = np.matrix([[1, 1, 1], [0, 0.34414, 1.772], [1.402, -0.71414, 0]])
    rgb2yuv_matrix = yuv2rgb_matrix.I
    full_cutoff = [0.0, 0.5, 0.5]
    yuvData = np.array(np.dot(rgb.reshape((width * height, 3)), rgb2yuv_matrix) + full_cutoff).reshape((width, height, 3))
    return yuvData

# Function: Convert RGGB raw image to Fake Gray image 
def RGGB2Gray(img):
    return np.mean(img, 2)

def rgb2NV12(rgb):
    rows, cols, _ = rgb.shape
    yuv2rgb_matrix = np.matrix([[1, 1, 1], [0, -0.34414, 1.772], [1.402, -0.71414, 0]])
    rgb2yuv_matrix = yuv2rgb_matrix.I
    print(rgb2yuv_matrix)
    # pdb.set_trace()
    # full_cutoff = [0.0, 0.5, 0.5]
    full_cutoff = np.array([[0.0, 0.5, 0.5]])
    yuvData = np.array(np.dot(rgb.reshape((rows * cols, 3)), rgb2yuv_matrix) + full_cutoff).reshape((rows, cols, 3))
    Y = yuvData[:,:,0]
    U = yuvData[:,:,1]
    V = yuvData[:,:,2]
    shrunkU = (U[0: :2, 0::2] + U[1: :2, 0: :2] + U[0: :2, 1: :2] + U[1: :2, 1: :2]) * 0.25
    shrunkV = (V[0: :2, 0::2] + V[1: :2, 0: :2] + V[0: :2, 1: :2] + V[1: :2, 1: :2]) * 0.25

    UV = np.zeros((rows//2, cols))
    ########################
    UV[:, 0 : :2] = shrunkU
    UV[:, 1 : :2] = shrunkV

    NV12 = np.vstack((Y, UV))
    print(NV12.shape)
    # pdb.set_trace()
    return yuvData, NV12

def yuv2rgb(yuv):
    rgb = yuv
    rgb[:,:,0] = yuv[:,:,0] + 1.402   * (yuv[:,:,2]-0.5)
    rgb[:,:,1] = yuv[:,:,0] - 0.34414 * (yuv[:,:,1]-0.5) - 0.71414*(yuv[:,:,2]-0.5)
    rgb[:,:,2] = yuv[:,:,0] + 1.772   * (yuv[:,:,1]-0.5)
    return rgb

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def save_rgb(file, data_y, data_u, data_v, bit = 10):
    img_yuv = np.stack((data_y, data_u, data_v), axis=2)
    img_yuv = img_yuv.astype(np.uint16)
    if bit == 10:
        img_yuv[img_yuv > 1023] = 1023
        img_yuv[img_yuv < 0] = 0
        img_yuv = img_yuv * 64
    elif bit == 12:
        img_yuv[img_yuv > 4095] = 4095
        img_yuv[img_yuv < 0] = 0
        img_yuv = img_yuv * 16
    elif bit == 14:
        img_yuv[img_yuv > 16383] = 16383
        img_yuv[img_yuv < 0] = 0
        img_yuv = img_yuv
    # pdb.set_trace()
    cv2.imwrite(file, img_yuv)
    print(file + ' saved')

def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output

def single_forward_google(model, inp, nmap):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp, nmap)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output

def single_forward_google_debug(model, inp, nmap):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output, gate_output = model(inp, nmap)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    gate_output = gate_output.data.float().cpu()
    return output, gate_output

def print_model_parm_flops(net, input_size, input_num=1, cuda=False):
    from torch.autograd import Variable
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(net)
    fn, c, h, w = input_size
    if input_num == 1:
        input = Variable(torch.rand(fn, c, h, w).unsqueeze(0).float(), requires_grad=True)
        if cuda:
            input = input.cuda()
        out = net(input)[0]
    else:
        input = []
        for i in range(input_num):
            input.append(Variable(torch.rand(c, h, w).unsqueeze(0), requires_grad=True))
            if cuda:
                input = [x in input, x.cuda()]
        if input_num == 2:
            out = net(input[0], input[1])[0]
        elif input_num == 3:
            out = net(input[0], input[1], input[2])[0]
        else:
            raise Exception("add {} input support".format(input_num))

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.4fG' % (total_flops / 1024.0 / 1024 / 1024))

def get_network_description(network):
        """Get the string and total parameters of the network"""
        # pdb.set_trace()
        # network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1, )))
    output_f = output_f + torch.flip(output, (-1, ))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2, )))
    output_f = output_f + torch.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


####################
# metric
####################
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

def takeFirst(elem):
    return elem[0]

def cal_lr_fea(fea, DW_model):
    B, N, C, H, W = fea.size()
    fea = fea.view(-1, C, H, W)
    LR_fea = DW_model(fea)
    LR_fea = LR_fea.view(B, N, LR_fea.shape[1], LR_fea.shape[2], LR_fea.shape[3])
    return LR_fea
    
def search_patch_NCC_2d_pymaid(image_patch, nmpa_patch, imgs_in_pad, img_in_nmap_pad, \
    start_x, start_y, small_scale, search_region):
    B, N, C, PsizeH, PsizeW = image_patch.shape
    _, _, _, H, W = imgs_in_pad.shape
    center_idx = N//2
    ncc_func = NormalizedCrossCorrelation(return_map=False,reduction='mean')
    ## recreat output
    image_patch_new = image_patch.clone()
    nmpa_patch_new = nmpa_patch.clone()

    ## downsampling the image patches
    # scale = 8
    scale = small_scale
    image_patch_small = torch.reshape(image_patch, (B*N,C,PsizeH,PsizeW))
    image_patch_small = F.interpolate(image_patch_small, scale_factor=1/scale, mode='bilinear', align_corners=False)
    imgs_in_pad_small = torch.reshape(imgs_in_pad, (B*N,C,H,W)) 
    imgs_in_pad_small = F.interpolate(imgs_in_pad_small, scale_factor=1/scale, mode='bilinear', align_corners=False)
    _,_,newPsizeH,newPsizeW = image_patch_small.shape
    _,_,newH,newW = imgs_in_pad_small.shape
    image_patch_small = torch.reshape(image_patch_small,(B, N, C, newPsizeH, newPsizeW))
    imgs_in_pad_small = torch.reshape(imgs_in_pad_small,(B, N, C, newH, newW))
    #search_region = int(min(newH, newW)/10)
    start_x = int(start_x/scale)
    start_y = int(start_y/scale)
    center_frame = image_patch_small[:,center_idx,:,:,:].clone()
    thr = -5
    # cadicate_idx_all = []
    for batch in range(B):
        start_x_current = start_x
        start_y_current = start_y
        # backfowd to the first frame
        for fr in range(center_idx-1,-1,-1):
            # print(fr)
            if fr != center_idx:
                step = 2
                cadicate_idx = cal_candidate_idx(search_region, step, start_x_current, start_y_current, batch, \
                    fr, newH, newW, imgs_in_pad_small, center_frame, newPsizeH, ncc_func)

                new_start_x = int(cadicate_idx[0][1])
                new_start_y = int(cadicate_idx[0][2])
                search_region_small = step
                # if cadicate_idx[0][0] > 0.6:
                cadicate_idx = cal_candidate_idx(search_region_small, 1, new_start_x, new_start_y, batch, \
                    fr, newH, newW, imgs_in_pad_small, center_frame, newPsizeH, ncc_func)

                # pdb.set_trace()
                # cadicate_idx_all.append(cadicate_idx)
                if len(cadicate_idx)>0:
                    if cadicate_idx[0][0] > thr:
                        nearest_x = int(cadicate_idx[0][1]*scale)
                        nearest_y = int(cadicate_idx[0][2]*scale)
                        start_x_current = int(cadicate_idx[0][1])
                        start_y_current = int(cadicate_idx[0][2])
                    else:
                        nearest_x = int(start_x*scale)
                        nearest_y = int(start_y*scale)
                    
                    image_patch_new[batch,fr,...] = \
                        imgs_in_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()
                    nmpa_patch_new[batch,fr,...] = \
                        img_in_nmap_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()

        # forward to the last frame
        start_x_current = start_x
        start_y_current = start_y
        for fr in range(center_idx+1,N):
            # print(fr)
            if fr != center_idx:
                step = 2
                cadicate_idx = cal_candidate_idx(search_region, step, start_x_current, start_y_current, batch, \
                    fr, newH, newW, imgs_in_pad_small, center_frame, newPsizeH, ncc_func)
                
                new_start_x = int(cadicate_idx[0][1])
                new_start_y = int(cadicate_idx[0][2])
                search_region_small = step
                # if cadicate_idx[0][0] > 0.6:
                cadicate_idx = cal_candidate_idx(search_region_small, 1, new_start_x, new_start_y, batch, \
                    fr, newH, newW, imgs_in_pad_small, center_frame, newPsizeH, ncc_func)

                # pdb.set_trace()
                # cadicate_idx_all.append(cadicate_idx)
                if len(cadicate_idx)>0:
                    if cadicate_idx[0][0] > thr:
                        nearest_x = int(cadicate_idx[0][1]*scale)
                        nearest_y = int(cadicate_idx[0][2]*scale)
                        start_x_current = int(cadicate_idx[0][1])
                        start_y_current = int(cadicate_idx[0][2])
                    else:
                        nearest_x = int(start_x*scale)
                        nearest_y = int(start_y*scale)
                    
                    image_patch_new[batch,fr,...] = \
                        imgs_in_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()
                    nmpa_patch_new[batch,fr,...] = \
                        img_in_nmap_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()

    # pdb.set_trace()
    return image_patch_new, nmpa_patch_new

def search_patch_NCC_2d_pymaid_wDSNet(image_patch, nmpa_patch, imgs_in_pad, img_in_nmap_pad, \
    lr_features,\
    start_x, start_y, small_scale, search_region):
    B, N, C, PsizeH, PsizeW = image_patch.shape
    _, _, _, H, W = imgs_in_pad.shape
    center_idx = N//2
    ncc_func = NormalizedCrossCorrelation(return_map=False,reduction='mean')
    #----- recreat output -----
    image_patch_new = image_patch.clone()
    nmpa_patch_new = nmpa_patch.clone()

    #----- select feature patch -----
    scale = small_scale
    start_x = int(start_x/scale)
    start_y = int(start_y/scale)
    center_feature = lr_features[:, center_idx, :, \
        start_x:start_x+PsizeH//scale, \
        start_y:start_y+PsizeW//scale].clone()

    ## downsampling the image patches
    _,_,newPsizeH,newPsizeW = center_feature.shape
    _,_,_,newH,newW = lr_features.shape
    thr = -5
    cadicate_idx_all = []
    for batch in range(B):
        start_x_current = start_x
        start_y_current = start_y
        # backfowd to the first frame
        for fr in range(center_idx-1,-1,-1):
            if fr != center_idx:
                step = 2
                cadicate_idx = cal_candidate_idx_wDSNet(search_region, step, start_x_current, start_y_current, batch, \
                    fr, newH, newW, lr_features, center_feature, newPsizeH, ncc_func)

                new_start_x = int(cadicate_idx[0][1])
                new_start_y = int(cadicate_idx[0][2])
                search_region_small = step
                cadicate_idx = cal_candidate_idx_wDSNet(search_region_small, 1, new_start_x, new_start_y, batch, \
                    fr, newH, newW, lr_features, center_feature, newPsizeH, ncc_func)

                cadicate_idx_all.append(cadicate_idx)
                if len(cadicate_idx)>0:
                    if cadicate_idx[0][0] > thr:
                        nearest_x = int(cadicate_idx[0][1]*scale)
                        nearest_y = int(cadicate_idx[0][2]*scale)
                        start_x_current = int(cadicate_idx[0][1])
                        start_y_current = int(cadicate_idx[0][2])
                    else:
                        nearest_x = int(start_x*scale)
                        nearest_y = int(start_y*scale)
                    
                    image_patch_new[batch,fr,...] = \
                        imgs_in_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()
                    nmpa_patch_new[batch,fr,...] = \
                        img_in_nmap_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()

        # forward to the last frame
        start_x_current = start_x
        start_y_current = start_y
        for fr in range(center_idx+1,N):
            # print(fr)
            if fr != center_idx:
                step = 2
                cadicate_idx = cal_candidate_idx_wDSNet(search_region, step, start_x_current, start_y_current, batch, \
                    fr, newH, newW, lr_features, center_feature, newPsizeH, ncc_func)
                
                new_start_x = int(cadicate_idx[0][1])
                new_start_y = int(cadicate_idx[0][2])
                search_region_small = step
                cadicate_idx = cal_candidate_idx_wDSNet(search_region_small, 1, new_start_x, new_start_y, batch, \
                    fr, newH, newW, lr_features, center_feature, newPsizeH, ncc_func)

                cadicate_idx_all.append(cadicate_idx)
                if len(cadicate_idx)>0:
                    if cadicate_idx[0][0] > thr:
                        nearest_x = int(cadicate_idx[0][1]*scale)
                        nearest_y = int(cadicate_idx[0][2]*scale)
                        start_x_current = int(cadicate_idx[0][1])
                        start_y_current = int(cadicate_idx[0][2])
                    else:
                        nearest_x = int(start_x*scale)
                        nearest_y = int(start_y*scale)
                    
                    image_patch_new[batch,fr,...] = \
                        imgs_in_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()
                    nmpa_patch_new[batch,fr,...] = \
                        img_in_nmap_pad[batch,fr,:,nearest_x:nearest_x+PsizeH,nearest_y:nearest_y+PsizeW].clone()

    # pdb.set_trace()
    return image_patch_new, nmpa_patch_new  

def search_patch_NCC_2d_pymaid_wDSNet_wE2E(image_patch, nmpa_patch, imgs_in_pad, img_in_nmap_pad, \
    lr_features,\
    start_x, start_y, small_scale, search_region,\
    gt_patch_raw, gt_raws, opti_step):
    B, N, C, PsizeH, PsizeW = image_patch.shape
    _, _, _, H, W = imgs_in_pad.shape
    center_idx = N//2
    ncc_func = NormalizedCrossCorrelation(return_map=False,reduction='mean')
    #----- recreat output -----
    image_patch_new = image_patch.clone()
    nmpa_patch_new = nmpa_patch.clone()
    ncc_scores_new = []
    ncc_scores_nor_new = []
    one_hot_gt = []
    # pdb.set_trace()
    # ----- resize image patch ------
    ## Using to calculate in patch level
    ## downsampling the image patches
    scale = small_scale
    image_patch_small = torch.reshape(gt_patch_raw, (B*N,C,PsizeH,PsizeW))
    image_patch_small = F.interpolate(image_patch_small, scale_factor=1/scale, mode='bilinear', align_corners=False)
    imgs_in_pad_small = torch.reshape(gt_raws, (B*N,C,H,W)) 
    imgs_in_pad_small = F.interpolate(imgs_in_pad_small, scale_factor=1/scale, mode='bilinear', align_corners=False)
    _,_,newPsizeH,newPsizeW = image_patch_small.shape
    _,_,newH,newW = imgs_in_pad_small.shape
    image_patch_small = torch.reshape(image_patch_small,(B, N, C, newPsizeH, newPsizeW))
    imgs_in_pad_small = torch.reshape(imgs_in_pad_small,(B, N, C, newH, newW))
    center_frame = image_patch_small[:,center_idx,:,:,:].clone()
    #----- select feature patch -----
    scale = small_scale
    start_x = int(start_x/scale)
    start_y = int(start_y/scale)
    center_feature = lr_features[:, center_idx, :, \
        start_x:start_x+PsizeH//scale, \
        start_y:start_y+PsizeW//scale].clone()

    ## downsampling the image patches
    # scale = 8
    _,_,newPsizeH,newPsizeW = center_feature.shape
    _,_,_,newH,newW = lr_features.shape
    thr = -5

    for batch in range(B):
        start_x_current = start_x
        start_y_current = start_y
        # backfowd to the first frame
        for fr in range(center_idx-1,-1,-1):
            # print(fr)
            if fr != center_idx:
                step = 2
                select_patch, select_np_patch, ncc_scores_temp, ncc_scores_nor_temp, one_hot_temp = \
                    cal_candidate_idx_wDSNet_E2E(search_region, step, \
                    start_x_current, start_y_current, batch, \
                    fr, newH, newW, lr_features, center_feature, \
                    newPsizeH, imgs_in_pad, img_in_nmap_pad, scale, ncc_func, \
                    imgs_in_pad_small, image_patch_small, center_frame, opti_step)

                image_patch_new[batch,fr,...] = select_patch.clone()
                nmpa_patch_new[batch,fr,...] = select_np_patch.clone()
                ncc_scores_new.append(ncc_scores_temp)
                ncc_scores_nor_new.append(ncc_scores_nor_temp)
                one_hot_gt.append(one_hot_temp)

        # forward to the last frame
        start_x_current = start_x
        start_y_current = start_y
        for fr in range(center_idx+1,N):
            # print(fr)
            if fr != center_idx:
                step = 2
                select_patch, select_np_patch, ncc_scores_temp, ncc_scores_nor_temp, one_hot_temp = \
                    cal_candidate_idx_wDSNet_E2E(search_region, step, \
                    start_x_current, start_y_current, batch, \
                    fr, newH, newW, lr_features, center_feature, \
                    newPsizeH, imgs_in_pad, img_in_nmap_pad, scale, ncc_func, \
                    imgs_in_pad_small, image_patch_small, center_frame, opti_step)

                image_patch_new[batch,fr,...] = select_patch.clone()
                nmpa_patch_new[batch,fr,...] = select_np_patch.clone()
                ncc_scores_new.append(ncc_scores_temp)
                ncc_scores_nor_new.append(ncc_scores_nor_temp)
                one_hot_gt.append(one_hot_temp)

    return image_patch_new, nmpa_patch_new, ncc_scores_new, ncc_scores_nor_new, one_hot_gt

def cal_candidate_idx(search_region, step, start_x, start_y, batch, fr, newH, newW, imgs_in, patch_in, patch_size, ncc_func):
    cadicate_idx = []
    center_patch_all = []
    candi_patch_all = []
    offset_all = []
    for x_offset in range(-search_region, search_region, step):
        x_temp = start_x + x_offset
        x_temp_end = start_x + x_offset + patch_size
        if x_temp<0 or x_temp_end>=newH:
            continue
        for y_offset in range(-search_region, search_region, step):
            y_temp = start_y + y_offset
            y_temp_end = start_y + y_offset + patch_size
            if y_temp<0 or y_temp_end>=newW:
                continue
            patch_temp = imgs_in[batch,fr:fr+1,:,x_temp:x_temp_end,y_temp:y_temp_end]
            # patch_temp = torch.mean(patch_temp,dim=1,keepdim=True)#.pow(1/2.2)
            candi_patch_all.append(patch_temp)

            center_frame_temp = patch_in[batch:batch+1,:,:,:]
            # center_frame_temp = torch.mean(center_frame_temp,dim=1,keepdim=True)#.pow(1/2.2)
            center_patch_all.append(center_frame_temp)
            offset_all.append(np.array([x_temp, y_temp]))
    
    ## Process in batch
    candi_patch_all = torch.cat(candi_patch_all, 0)
    center_patch_all = torch.cat(center_patch_all, 0)
    candi_patch_all = candi_patch_all.cuda()
    center_patch_all = center_patch_all.cuda()
    all_fea_candi = candi_patch_all
    all_fea_center = center_patch_all
    # Calculate normalized correlation between patches
    ncc_scores = ncc_func(all_fea_center.contiguous(), all_fea_candi.contiguous()) - 1
    # ncc_scores = -torch.mean(torch.mean(torch.mean(torch.abs(all_fea_center.half()-all_fea_candi.half()),1),1),1)

    ncc_scores = ncc_scores.float().cpu().detach().numpy()
    offset_all_temp = np.stack(offset_all, axis=0)
    ncc_scores = np.expand_dims(ncc_scores, axis=1)
    cadicate_idx = np.concatenate((ncc_scores,offset_all_temp), axis=1)
    # Sort cadicate idx
    cadicate_idx = cadicate_idx.tolist()
    cadicate_idx.sort(key=takeFirst, reverse=True)
    return cadicate_idx

def cal_candidate_idx_wDSNet(search_region, step, start_x, start_y, batch, 
                            fr, newH, newW, imgs_in, patch_in, patch_size, ncc_func):
    cadicate_idx = []
    center_patch_all = []
    candi_patch_all = []
    offset_all = []
    index = 0
    for x_offset in range(-search_region, search_region, step):
        x_temp = start_x + x_offset
        x_temp_end = start_x + x_offset + patch_size
        if x_temp<0 or x_temp_end>=newH:
            continue
        for y_offset in range(-search_region, search_region, step):
            y_temp = start_y + y_offset
            y_temp_end = start_y + y_offset + patch_size
            if y_temp<0 or y_temp_end>=newW:
                continue
            patch_temp = imgs_in[batch,fr:fr+1,:,x_temp:x_temp_end,y_temp:y_temp_end]
            candi_patch_all.append(patch_temp)

            center_frame_temp = patch_in[batch:batch+1,:,:,:]
            center_patch_all.append(center_frame_temp)
            offset_all.append(np.array([x_temp, y_temp]))
            index = index + 1
    
    ## Process in batch
    candi_patch_all = torch.cat(candi_patch_all, 0)
    center_patch_all = torch.cat(center_patch_all, 0)
    candi_patch_all = candi_patch_all.cuda()
    center_patch_all = center_patch_all.cuda()
    all_fea_candi = candi_patch_all
    all_fea_center = center_patch_all
    # Calculate normalized correlation between patches
    ncc_scores = -torch.mean(torch.mean(torch.mean(torch.abs(all_fea_center-all_fea_candi),1),1),1)
    ncc_scores = F.normalize(ncc_scores, dim=0)
    ncc_scores = ncc_scores.float().cpu().detach().numpy()
    offset_all_temp = np.stack(offset_all, axis=0)
    ncc_scores = np.expand_dims(ncc_scores, axis=1)
    cadicate_idx = np.concatenate((ncc_scores,offset_all_temp), axis=1)
    # Sort cadicate idx
    cadicate_idx = cadicate_idx.tolist()
    cadicate_idx.sort(key=takeFirst, reverse=True)
    return cadicate_idx

def cal_candidate_idx_wDSNet_E2E(search_region, step, start_x, start_y, batch, fr, \
    newH, newW, imgs_in, patch_in, patch_size, img_full_size, \
    noise_map_full_size, scale, ncc_func, \
    imgs_in_pad_small, image_patch_small, center_frame, opti_step):
    cadicate_idx = []
    center_patch_all = []
    candi_patch_all = []
    candi_patch_all_full = []
    candi_nm_patch_all_full = []
    # ------ Cal one-hot GT vector -----
    center_patch_img_all = []
    candi_patch_img_all = []

    offset_all = []
    for x_offset in range(-search_region, search_region, step):
        x_temp = start_x + x_offset
        x_temp_end = start_x + x_offset + patch_size
        if x_temp<0 or x_temp_end>=newH:
            continue
        for y_offset in range(-search_region, search_region, step):
            y_temp = start_y + y_offset
            y_temp_end = start_y + y_offset + patch_size
            if y_temp<0 or y_temp_end>=newW:
                continue
            patch_temp = imgs_in[batch,fr:fr+1,:,x_temp:x_temp_end,y_temp:y_temp_end]
            patch_temp_full = img_full_size[batch,fr:fr+1,:,x_temp*scale:x_temp_end*scale,\
                y_temp*scale:y_temp_end*scale]
            patch_np_temp_full = noise_map_full_size[batch,fr:fr+1,:,x_temp*scale:x_temp_end*scale,\
                y_temp*scale:y_temp_end*scale]
            candi_patch_all.append(patch_temp)
            candi_patch_all_full.append(patch_temp_full)
            candi_nm_patch_all_full.append(patch_np_temp_full)

            center_frame_temp = patch_in[batch:batch+1,:,:,:]
            center_patch_all.append(center_frame_temp)
            offset_all.append(np.array([x_temp, y_temp]))

            # ------ Cal one-hot GT vector -----
            patch_img_temp = imgs_in_pad_small[batch,fr:fr+1,:,x_temp:x_temp_end,y_temp:y_temp_end]
            candi_patch_img_all.append(patch_img_temp)
            center_img_temp = center_frame[batch:batch+1,:,:,:]
            center_patch_img_all.append(center_img_temp)

    ## Process in batch
    candi_patch_all = torch.cat(candi_patch_all, 0)
    candi_patch_all_full = torch.cat(candi_patch_all_full, 0)
    candi_nm_patch_all_full = torch.cat(candi_nm_patch_all_full, 0)
    center_patch_all = torch.cat(center_patch_all, 0)
    candi_patch_all = candi_patch_all.cuda()
    candi_patch_all_full = candi_patch_all_full.cuda()
    candi_nm_patch_all_full = candi_nm_patch_all_full.cuda()
    center_patch_all = center_patch_all.cuda()
    all_fea_candi = candi_patch_all
    all_fea_center = center_patch_all
    # ------ Cal one-hot GT vector -----
    candi_patch_img_all = torch.cat(candi_patch_img_all, 0)
    center_patch_img_all = torch.cat(center_patch_img_all, 0)
    candi_patch_img_all = candi_patch_img_all.cuda()
    center_patch_img_all = center_patch_img_all.cuda()
    #ncc_scores_gt = ncc_func(center_patch_img_all.contiguous(), \
    #    candi_patch_img_all.contiguous()) - 1
    ncc_scores_gt = -torch.mean(torch.mean(torch.mean(torch.abs(center_patch_img_all-candi_patch_img_all),1),1),1)
    ncc_scores_gt = F.normalize(ncc_scores_gt, dim=0)

    # Calculate normalized correlation between patches
    ncc_scores = -torch.mean(torch.mean(torch.mean(torch.abs(all_fea_center-all_fea_candi),1),1),1)
    ncc_scores = F.normalize(ncc_scores, dim=0)
    # ----- Differentiable KNN (top-1) -----
    temperature = 1e-2
    # print(opti_step)
    if opti_step > 300000:
        temperature = 1e-3
       
    ncc_score_wT = ncc_scores/temperature
    ncc_score_wT_nor = F.softmax(ncc_score_wT)

    ncc_score_wT_nor = ncc_score_wT_nor.view(-1, 1, 1, 1).float()

    select_patch = candi_patch_all_full * ncc_score_wT_nor
    select_patch = torch.sum(select_patch, 0)
    select_np_patch = candi_nm_patch_all_full * ncc_score_wT_nor
    select_np_patch = torch.sum(select_np_patch, 0)
    return select_patch, select_np_patch, ncc_scores, ncc_score_wT_nor, ncc_scores_gt

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

## used in test process:
def pad_img_2_setscale(img, need_scale):
    B, N, C, H_ori, W_ori = img.size()
    h_res = need_scale - H_ori % need_scale
    w_res = need_scale - W_ori % need_scale
    dim=(0,w_res,0,h_res)
    x_pad = img.clone()
    x_pad = F.pad(x_pad,dim,"constant",value=0)
    x_pad = x_pad.contiguous()
    _,_,_,H_new, W_new = x_pad.size()
    return x_pad, H_ori, W_ori, H_new, W_new

def caligned_wPBM(imgs_in, img_in_nmap, scale, test_patch_size, \
    patch_extend, search_region):
    H = imgs_in.shape[3]
    W = imgs_in.shape[4]
    patch_size = test_patch_size
    new_patch_size = test_patch_size + 2*patch_extend
    h_num = (H-2*patch_extend) // patch_size
    w_num = (W-2*patch_extend) // patch_size
    # pdb.set_trace()
    all_noise_patches = []
    all_nosie_maps = []
    for h_index in range(math.floor(h_num)):
        for w_index in range(math.floor(w_num)):
            start_x = h_index * patch_size
            end_x = start_x + new_patch_size - 1
            start_y = w_index * patch_size
            end_y = start_y + new_patch_size - 1
            
            imgs_in_patch = imgs_in[:,:,:,start_x:end_x+1,start_y:end_y+1]
            nmap_in_patch = img_in_nmap[:,:,:,start_x:end_x+1,start_y:end_y+1]
            # pdb.set_trace()
            image_patch_new, nmap_patch_new = \
                search_patch_NCC_2d_pymaid(imgs_in_patch, nmap_in_patch, \
                    imgs_in, img_in_nmap, \
                    start_x, start_y, scale, search_region)
            # pdb.set_trace()
            all_noise_patches.append(image_patch_new)
            all_nosie_maps.append(nmap_patch_new)
            
    all_noise_patches = torch.cat(all_noise_patches, 0) 
    all_nosie_maps = torch.cat(all_nosie_maps, 0)
    patch_num = all_noise_patches.shape[0]
    return all_noise_patches, all_nosie_maps, patch_num, h_num, w_num

def caligned_wPBM_wDSNet(imgs_in, img_in_nmap, scale, test_patch_size, \
    patch_extend, search_region, net_vgg):
    H = imgs_in.shape[3]
    W = imgs_in.shape[4]
    patch_size = test_patch_size
    new_patch_size = test_patch_size + 2*patch_extend
    h_num = (H-2*patch_extend) // patch_size
    w_num = (W-2*patch_extend) // patch_size
    # pdb.set_trace()
    all_noise_patches = []
    all_nosie_maps = []
    ## lr_features on full image
    lr_features = cal_lr_fea(imgs_in, net_vgg)
    for h_index in range(math.floor(h_num)):
        for w_index in range(math.floor(w_num)):
            start_x = h_index * patch_size
            end_x = start_x + new_patch_size - 1
            start_y = w_index * patch_size
            end_y = start_y + new_patch_size - 1
            
            imgs_in_patch = imgs_in[:,:,:,start_x:end_x+1,start_y:end_y+1]
            nmap_in_patch = img_in_nmap[:,:,:,start_x:end_x+1,start_y:end_y+1]
            # pdb.set_trace()
            image_patch_new, nmap_patch_new = \
                search_patch_NCC_2d_pymaid_wDSNet(imgs_in_patch, nmap_in_patch, \
                    imgs_in, img_in_nmap, lr_features,\
                    start_x, start_y, scale, search_region)
            
            # pdb.set_trace()
            all_noise_patches.append(image_patch_new)
            all_nosie_maps.append(nmap_patch_new)
            
    all_noise_patches = torch.cat(all_noise_patches, 0) 
    all_nosie_maps = torch.cat(all_nosie_maps, 0)
    patch_num = all_noise_patches.shape[0]
    return all_noise_patches, all_nosie_maps, patch_num, h_num, w_num

def caligned_wPBM_test(imgs_in, img_in_nmap, scale, test_patch_size, \
    patch_extend, search_region, net_down=None):
    H = imgs_in.shape[3]
    W = imgs_in.shape[4]
    patch_size = test_patch_size
    new_patch_size = test_patch_size + 2*patch_extend
    h_num = (H-2*patch_extend) // patch_size
    w_num = (W-2*patch_extend) // patch_size
    # pdb.set_trace()
    all_noise_patches = []
    all_nosie_maps = []
    if net_down is not None:
        ## vgg_features on full image
        lr_features = cal_lr_fea(imgs_in, net_down)
        
    for h_index in range(math.floor(h_num)):
        for w_index in range(math.floor(w_num)):
            start_x = h_index * patch_size
            end_x = start_x + new_patch_size - 1
            start_y = w_index * patch_size
            end_y = start_y + new_patch_size - 1
            
            imgs_in_patch = imgs_in[:,:,:,start_x:end_x+1,start_y:end_y+1]
            nmap_in_patch = img_in_nmap[:,:,:,start_x:end_x+1,start_y:end_y+1]
            # pdb.set_trace()
            image_patch_new, nmap_patch_new = \
                search_patch_NCC_2d_pymaid_wDSNet(imgs_in_patch, nmap_in_patch, \
                    imgs_in, img_in_nmap, lr_features,\
                    start_x, start_y, scale, search_region)
    
            all_noise_patches.append(image_patch_new)
            all_nosie_maps.append(nmap_patch_new)
            
    all_noise_patches = torch.cat(all_noise_patches, 0) 
    all_nosie_maps = torch.cat(all_nosie_maps, 0)
    patch_num = all_noise_patches.shape[0]
    return all_noise_patches, all_nosie_maps, patch_num, h_num, w_num

def batch_forward(model, all_img_in_patches, all_nmap_in_patches, patch_num, max_batch_num):
    ## process on batch patches
    batch_each = int(max_batch_num)
    iters_num = int(patch_num//batch_each)
    output_patches = []
    # pdb.set_trace()
    for i in range(iters_num):
        image_patch_temp = all_img_in_patches[batch_each*i:batch_each*(i+1),...]
        nmap_patch_temp = all_nmap_in_patches[batch_each*i:batch_each*(i+1),...]
        image_patch_temp = image_patch_temp.cuda()
        nmap_patch_temp = nmap_patch_temp.cuda()
        output = single_forward_google(model, image_patch_temp, nmap_patch_temp)
        output_patches.append(output)
    if (patch_num - iters_num*batch_each) > 0:
        image_patch_temp = all_img_in_patches[batch_each*iters_num:,...]
        nmap_patch_temp = all_nmap_in_patches[batch_each*iters_num:,...]
        image_patch_temp = image_patch_temp.cuda()
        nmap_patch_temp = nmap_patch_temp.cuda()
        output = single_forward_google(model, image_patch_temp, nmap_patch_temp)
        output_patches.append(output)
    output_patches = torch.cat(output_patches,0)
    return output_patches

def merge_back(output_patches, denoised_img, h_num, w_num, patch_size, patch_extend):
    patch_idx = 0
    for h_index in range(math.floor(h_num)):
        for w_index in range(math.floor(w_num)):
            output = output_patches[patch_idx:patch_idx+1,...]
            patch_idx = patch_idx + 1
                
            start_x2 = h_index * 2 * patch_size
            end_x2 = start_x2 + 2 * patch_size - 1
            start_y2 = w_index * 2 * patch_size
            end_y2 = start_y2 + 2 * patch_size - 1
            if patch_extend != 0:
                output = output[:, :, 2*patch_extend:-2*patch_extend, 2*patch_extend:-2*patch_extend]
            denoised_img[:, :, start_x2:end_x2+1, start_y2:end_y2+1] = output
        
    return denoised_img

def save_tensor_debug(debug_info):
    debug_folder = '../Debug/'
    # debug_info = [v.squeeze().float().cpu().numpy() for v in debug_info]
    debug_info = debug_info.squeeze().float().cpu().numpy()
    for fea_in in range(4):
        for fr_in in range(5):
            fwd_f = debug_info[fr_in, fea_in, :, :].copy()
            #fwd_f = np.abs(fwd_f)
            #fwd_f = fwd_f / np.max(fwd_f)
            fwd_f = ((fwd_f**(1/2.0)) * 255.0).round()
            fwd_f = fwd_f.astype(np.uint8)
            cv2.imwrite(osp.join(debug_folder, 'CH{}_Fnum{}.png'.format(fea_in,fr_in)), fwd_f)
    pdb.set_trace()

def crop_imgs(img, ratio):
    # if Run out of CUDA memory, you can crop images to small pieces by this function. Just set --crop_scale 4
    b, n, c, h, w = img.shape
    h_psize = h // ratio
    w_psize = w // ratio
    imgs = torch.zeros(ratio ** 2, n, c, h_psize, w_psize)

    for i in range(ratio):
        for j in range(ratio):
            imgs[i * ratio + j] = img[0, :, :, i * h_psize:(i + 1) * h_psize, j * w_psize:(j + 1) * w_psize]
    return imgs

def binning_imgs(img, ratio):
    patch_n, n, c, h, w = img.shape
    output = torch.zeros((1, n, c, h * ratio, w * ratio))

    for i in range(ratio):
        for j in range(ratio):
            output[:, :, :, i * h:(i + 1) * h, j * w:(j + 1) * w] = img[i * ratio + j]
    return output
