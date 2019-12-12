is_cuda = lambda m: next(m.parameters()).is_cuda

from datajoint.expression import QueryExpression
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from contextlib import contextmanager
import hashlib

from scipy import ndimage
from scipy import signal

import sys

from os import path
import copy
from neuro_data.static_images import transforms
from neuro_data.static_images.datasets import StaticImageSet

import cv2
import imageio
import matplotlib.pyplot as plt
from itertools import product, zip_longest
import seaborn as sns

# Utility functions to save and load datasets
def pack_transform(t):
    if isinstance(t, transforms.Normalizer):
        fields = ['buggy', 'normalize_per_image', 'exclude', 'stats_source']
        t_config = {f: getattr(t, f) for f in fields}
        t_config['type'] = 'Normalizer'
    elif isinstance(t, transforms.Subsample):
        t_config = dict(idx=t.idx)
        t_config['type'] = 'Subsample'
    elif isinstance(t, transforms.ToTensor):
        t_config = dict(cuda=t.cuda)
        t_config['type'] = 'ToTensor'
    else:
        raise ValueError('Packing transform type {} is not supported yet'.format(t.__class__.__name__))
    return t_config

def unpack_transform(dataset, t_config):
    t_type = t_config.pop('type')
    if t_type == 'Normalizer':
        return transforms.Normalizer(dataset, **t_config)
    elif t_type == 'Subsample':
        return transforms.Subsample(**t_config)
    elif t_type == 'ToTensor':
        return transforms.ToTensor(**t_config)

def pack_dataset(ds, filename=None):
    if filename is None:
        filename = ds._fid.filename
    configs = dict(filename=filename, data_keys=ds.data_keys, cache_raw=ds.cache_raw, stats_source=ds.stats_source)
    configs['transforms'] = [pack_transform(t) for t in ds.transforms]
    return configs

def unpack_dataset(ds_config, filename=None):
    ds_config = copy.deepcopy(ds_config)
    t_configs = ds_config.pop('transforms')
    data_keys = ds_config.pop('data_keys')
    saved_filename = ds_config.pop('filename')
    if filename is None:
        filename = saved_filename
    new_ds = StaticImageSet(filename, *data_keys, **ds_config)
    transforms = [unpack_transform(new_ds, t) for t in t_configs]
    new_ds.transforms.extend(transforms)
    return new_ds

def load_dataset(filename, ds_path=None):
    if ds_path is None:
        ds_path = path.join(path.split(filename)[0], 'dataset_config.pt')
    dataset_config = torch.load(ds_path)
    return unpack_dataset(dataset_config, filename)

def list_hash(values):
    """
    Returns MD5 digest hash values for a list of values
    """
    hashed = hashlib.md5()
    for v in values:
        hashed.update(str(v).encode())
    return hashed.hexdigest()


def key_hash(key):
    """
    32-byte hash used for lookup of primary keys of jobs
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()

class silence:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def safe_key(self):
    data = self.proj(*self.heading.non_blobs).fetch()
    return data[self.heading.primary_key]


def fetchdf(self, *args, **kwargs):
    return pd.DataFrame(self.fetch(*args, **kwargs))


QueryExpression.safe_key = safe_key
QueryExpression.fetchdf = fetchdf

from attorch.layers import SpatialTransformerPyramid2d, SpatialTransformerPooled2d
from attorch.constraints import positive
import torch.nn.functional as F


def disc_center_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    self.grid.data = torch.clamp(self.grid.data, -1, 1)
    N, c, h, w = x.size()
    m = self.gauss_pyramid.scale_n + 1
    feat = self.features.view(1, m * c, self.outdims)

    pools = []
    for xx in self.gauss_pyramid(x):
        N, ch, img_h, img_w = xx.size()
        ctr_h, ctr_w = img_h // 2, img_w // 2
        pools.append(xx[..., ctr_h, ctr_w].unsqueeze(-1).expand(N, ch, self.outdims))

    y = torch.cat(pools, dim=1)
    y = (y * feat).sum(1).view(N, self.outdims)

    if self.bias is not None:
        y = y + self.bias
    return y

def disc_center_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    self.grid.data = torch.clamp(self.grid.data, -1, 1)
    N, c, h, w = x.size()
    m = self.gauss_pyramid.scale_n + 1
    feat = self.features.view(1, m * c, self.outdims)

    pools = []
    for xx in self.gauss_pyramid(x):
        N, ch, img_h, img_w = xx.size()
        ctr_h, ctr_w = img_h // 2, img_w // 2
        pools.append(xx[..., ctr_h, ctr_w].unsqueeze(-1).expand(N, ch, self.outdims))

    y = torch.cat(pools, dim=1)
    y = (y * feat).sum(1).view(N, self.outdims)

    if self.bias is not None:
        y = y + self.bias
    return y

def simple_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    N, ch, h, w = x.size()
    feat = self.features.view(1, -1, self.outdims)
    feat = feat[:, 0:ch, :]
    ctr_h, ctr_w = h // 2, w // 2

    y = x[..., ctr_h, ctr_w].unsqueeze(-1).expand(N, ch, self.outdims)

    y = (y * feat).sum(1).view(N, self.outdims)

    if self.bias is not None:
        y = y + self.bias

    return y

def fixed_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    N, ch, h, w = x.size()
    feat = self.features.view(1, -1, self.outdims)
    feat_scale = feat.mean()
    ctr_h, ctr_w = h // 2, w // 2

    y = x[..., ctr_h, ctr_w].unsqueeze(-1).expand(N, ch, self.outdims)

    y = (y * feat_scale).sum(1).view(N, self.outdims)

    if self.bias is not None:
        y = y + self.bias

    return y


def trunc_center_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    self.grid.data = torch.clamp(self.grid.data, -1, 1)
    N, c, h, w = x.size()
    m = self.gauss_pyramid.scale_n + 1
    feat = self.features.view(1, m * c, self.outdims)
    start = self.scale_idx * c
    feat = feat[:, start:(start+c), :]

    pools = []

    for i, xx in enumerate(self.gauss_pyramid(x)):
        if i == self.scale_idx:
            N, ch, img_h, img_w = xx.size()
            ctr_h, ctr_w = img_h // 2, img_w // 2
            pools.append(xx[..., ctr_h, ctr_w].unsqueeze(-1).expand(N, ch, self.outdims))

    y = torch.cat(pools, dim=1)
    y = (y * feat).sum(1).view(N, self.outdims)

    if self.bias is not None:
        y = y + self.bias
    return y


def discretized_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    self.grid.data = torch.clamp(self.grid.data, -1, 1)
    N, c, h, w = x.size()
    m = self.gauss_pyramid.scale_n + 1
    feat = self.features.view(1, m * c, self.outdims)


    if shift is None:
        grid = self.grid.expand(N, self.outdims, 1, 2)
    else:
        grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]

    pools = []


    for xx in self.gauss_pyramid(x):
        _, _, img_h, img_w = xx.size()
        img_shape = torch.tensor([img_w, img_h]).type_as(grid)
        adj_grid = torch.round((grid + 1) / 2 * (img_shape - 1)) * 2 / (img_shape - 1) - 1
        pools.append(F.grid_sample(xx, adj_grid))

    y = torch.cat(pools, dim=1).squeeze(-1)
    y = (y * feat).sum(1).view(N, self.outdims)

    if self.bias is not None:
        y = y + self.bias
    return y

def indexed_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    self.grid.data = torch.clamp(self.grid.data, -1, 1)
    N, c, w, h = x.size()
    m = self.gauss_pyramid.scale_n + 1
    feat = self.features.view(1, m * c, self.outdims)


    if shift is None:
        grid = self.grid.expand(N, self.outdims, 1, 2)
    else:
        grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]


    pools = []

    for xx in self.gauss_pyramid(x):
        _, _, img_w, img_h = xx.size()

        img_indexer = torch.tensor([(img_w -1) * img_h, img_h - 1]).type_as(grid)
        (grid + 1) / 2
        img_index = torch.round((grid + 1) / 2 * (img_shape - 1))
        pools.append(F.grid_sample(xx, adj_grid))

    y = torch.cat(pools, dim=1).squeeze(-1)
    y = (y * feat).sum(1).view(N, self.outdims)

    if self.bias is not None:
        y = y + self.bias
    return y


# define alternate forward on SpatialTransformerPyramid2d
def all_forward(self, x, shift=None):
    if self.positive:
        positive(self.features)
    self.grid.data = torch.clamp(self.grid.data, -1, 1)
    N, c, w, h = x.size()
    m = self.gauss_pyramid.scale_n + 1
    feat = self.features.view(m * c, self.outdims, 1, 1).permute([1, 0, 2, 3]).contiguous()
    stacked = torch.cat(self.gauss_pyramid(x), dim=1)
    y = F.conv2d(stacked, feat, self.bias)
    _, _, yw, yh = y.size()

    return y

def center_forward(self, x, shift=None):
    grid_backup = self.grid.data.clone()
    self.grid.data.zero_()
    output = self.old_forward(x, shift=shift)
    self.grid.data = grid_backup
    return output



@contextmanager
def discrete_readout():
    print('Using discretized forward on SpatialTransformerPyramid2d')
    old_forward = SpatialTransformerPyramid2d.forward
    SpatialTransformerPyramid2d.forward = discretized_forward
    yield
    print('Restoring normal forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.forward = old_forward

@contextmanager
def simple_readout():
    print('Using simplified forward on SpatialTransformerPyramid2d')
    old_forward = SpatialTransformerPyramid2d.forward
    SpatialTransformerPyramid2d.forward = simple_forward
    yield
    print('Restoring normal forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.forward = old_forward


@contextmanager
def fixed_readout():
    print('Using fixed forward on SpatialTransformerPyramid2d')
    old_forward = SpatialTransformerPyramid2d.forward
    SpatialTransformerPyramid2d.forward = fixed_forward
    yield
    print('Restoring normal forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.forward = old_forward


@contextmanager
def disc_center_readout():
    print('Using discrete centered forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.old_forward = SpatialTransformerPyramid2d.forward
    SpatialTransformerPyramid2d.forward = disc_center_forward
    yield
    print('Restoring normal forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.forward = SpatialTransformerPyramid2d.old_forward

@contextmanager
def trunc_center_readout(scale_idx=0):
    print('Using truncated centered forward with scale={} on SpatialTransformerPyramid2d'.format(scale_idx))
    SpatialTransformerPyramid2d.old_forward = SpatialTransformerPyramid2d.forward
    SpatialTransformerPyramid2d.forward = trunc_center_forward
    SpatialTransformerPyramid2d.scale_idx = scale_idx
    yield
    print('Restoring normal forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.forward = SpatialTransformerPyramid2d.old_forward

@contextmanager
def alt_forward():
    print('Using alternate forward on SpatialTransformerPyramid2d')
    old_forward = SpatialTransformerPyramid2d.forward
    SpatialTransformerPyramid2d.forward = all_forward
    yield
    print('Restoring normal forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.forward = old_forward


@contextmanager
def center_readout():
    print('Using centeralized forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.old_forward = SpatialTransformerPyramid2d.forward
    SpatialTransformerPyramid2d.forward = center_forward
    yield
    print('Restoring normal forward on SpatialTransformerPyramid2d')
    SpatialTransformerPyramid2d.forward = SpatialTransformerPyramid2d.old_forward




SpatialTransformerPyramid2d.alt_forward = alt_forward
SpatialTransformerPyramid2d.center_readout = center_readout
SpatialTransformerPyramid2d.disc_center_readout = disc_center_readout
SpatialTransformerPyramid2d.discrete_readout = discrete_readout
SpatialTransformerPyramid2d.trunc_center_readout = trunc_center_readout
SpatialTransformerPyramid2d.simple_readout = simple_readout
SpatialTransformerPyramid2d.fixed_readout = fixed_readout


# # define alternate forward on SpatialTransformerPooled2d
#
# def forward(self, x, shift=None):
#     if self.positive:
#         positive(self.features)
#     self.grid.data = torch.clamp(self.grid.data, -1, 1)
#     N, c, w, h = x.size()
#     m = self.pool_steps + 1
#     feat = self.features.view(1, m * c, self.outdims)
#
#     if shift is None:
#         grid = self.grid.expand(N, self.outdims, 1, 2)
#     else:
#         grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]
#
#     pools = [F.grid_sample(x, grid)]
#     for _ in range(self.pool_steps):
#         x = self.avg(x)
#         pools.append(F.grid_sample(x, grid))
#     y = torch.cat(pools, dim=1)
#     y = (y.squeeze(-1) * feat).sum(1).view(N, self.outdims)
#
#     if self.bias is not None:
#         y = y + self.bias
#     return y



def fft_smooth(grad, factor=1/4):
    """
    Tones down the gradient with 1/sqrt(f) filter in the Fourier domain.
    Equivalent to low-pass filtering in the spatial domain.
    """
    if factor == 0:
        return grad
    #h, w = grad.size()[-2:]
    # grad = tf.transpose(grad, [0, 3, 1, 2])
    # grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    h, w = grad.size()[-2:]
    # grad = tf.transpose(grad, [0, 3, 1, 2])
    # grad_fft = tf.fft2d(tf.cast(grad, tf.complex64))
    tw = np.minimum(np.arange(0, w), np.arange(w-1, -1, -1), dtype=np.float32)  # [-(w+2)//2:]
    th = np.minimum(np.arange(0, h), np.arange(h-1, -1, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** (factor))
    F = grad.new_tensor(t / t.mean()).unsqueeze(-1)
    pp = torch.rfft(grad.data, 2, onesided=False)
    return torch.irfft(pp * F, 2, onesided=False)


def blur(img, sigma):
    if sigma > 0:
        for d in range(len(img)):
            img[d] = ndimage.filters.gaussian_filter(img[d], sigma, order=0)
    return img


def blur_in_place(tensor, sigma):
    blurred = np.stack([blur(im, sigma) for im in tensor.cpu().numpy()])
    tensor.copy_(torch.Tensor(blurred))


def named_forward(self, input, name=None):
    for mod_name, module in self._modules.items():
        input = module(input)
        if mod_name == name:
            return input
    return input


nn.Sequential.forward = named_forward


def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def process(x, mu=0.4, sigma=0.224):
    """ Normalize and move channel dim in front of height and width"""
    x = (x - mu) / sigma
    if isinstance(x, torch.Tensor):
        return x.transpose(-1, -2).transpose(-2, -3)
    else:
        return np.moveaxis(x, -1, -3)


def unprocess(x, mu=0.4, sigma=0.224):
    """Inverse of process()"""
    x = x * sigma + mu
    if isinstance(x, torch.Tensor):
        return x.transpose(-3, -2).transpose(-2, -1)
    else:
        return np.moveaxis(x, -3, -1)


def storeoutput(self, input, output):
    print(self.__class__.__name__)
    self._output = output
    raise Exception('Truncation')


def batch_mean(batch, keepdim=False):
    """ Compute mean for a batch of images. """
    mean = batch.view(len(batch), -1).mean(-1)
    if keepdim:
        mean = mean.view(len(batch), 1, 1, 1)
    return mean


def batch_std(batch, keepdim=False, unbiased=True):
    """ Compute std for a batch of images. """
    std = batch.view(len(batch), -1).std(-1, unbiased=unbiased)
    if keepdim:
        std = std.view(len(batch), 1, 1, 1)
    return std


def make_step(net, src, step_size=1.5, sigma=None, precond=0, step_gain=1,
              blur=True, jitter=0, eps=1e-12, clip=True, bias=0.4, scale=0.224,
              train_norm=None, norm=None, add_loss=0, _eps=1e-12):
    """ Update src in place making a gradient ascent step in the output of net.

    Arguments:
        net (nn.Module or function): A backpropagatable function/module that receives
            images in (B x C x H x W) form and outputs a scalar value per image.
        src (torch.Tensor): Batch of images to update (B x C x H x W).
        step_size (float): Step size to use for the update: (im_old += step_size * grad)
        sigma (float): Standard deviation for gaussian smoothing (if used, see blur).
        precond (float): Strength of gradient smoothing.
        step_gain (float): Scaling factor for the step size.
        blur (boolean): Whether to blur the image after the update.
        jitter (int): Randomly shift the image this number of pixels before forwarding
            it through the network.
        eps (float): Small value to avoid division by zero.
        clip (boolean): Whether to clip the range of the image to be in [0, 255]
        train_norm (float): Decrease standard deviation of the image feed to the
            network to match this norm. Expressed in original pixel values. Unused if
            None
        norm (float): Decrease standard deviation of the image to match this norm after
            update. Expressed in z-scores. Unused if None
        add_loss (function): An additional term to add to the network activation before
            calling backward on it. Usually, some regularization.
    """
    if src.grad is not None:
        src.grad.zero_()

    # apply jitter shift
    if jitter > 0:
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)  # use uniform distribution
        ox, oy = int(ox), int(oy)
        src.data = roll(roll(src.data, ox, -1), oy, -2)

    img = src
    if train_norm is not None and train_norm > 0.0:
        # normalize the image in backpropagatable manner
        img_idx = batch_std(src.data) + _eps > train_norm / scale  # images to update
        if img_idx.any():
            img = src.clone() # avoids overwriting original image but lets gradient through
            img[img_idx] = ((src[img_idx] / (batch_std(src[img_idx], keepdim=True) +
                                             _eps)) * (train_norm / scale))

    y = net(img)
    (y.mean() + add_loss).backward()

    grad = src.grad
    if precond > 0:
        grad = fft_smooth(grad, precond)

    # src.data += (step_size / (batch_mean(torch.abs(grad.data), keepdim=True) + eps)) * (step_gain / 255) * grad.data
    src.data += (step_size / (torch.abs(grad.data).mean() + eps)) * (step_gain / 255) * grad.data
    # * both versions are equivalent for a single-image batch, for batches with more than
    # one image the first one is better but it drawns out the gradients that are spatially
    # wide; for instance a gradient of size 5 x 5 pixels all at amplitude 1 will produce a
    # higher change in an image of the batch than a gradient of size 20 x 20 all at
    # amplitude 1 in another. This is alright in most cases, but when generating diverse
    # images with min linkage (i.e, all images receive gradient from the signal and two
    # get the gradient from the diversity term) it drawns out the gradient generated from
    # the diversity term (because it is usually bigger spatially than the signal gradient)
    # and becomes hard to find very diverse images (i.e., increasing the diversity term
    # has no effect because the diversity gradient gets rescaled down to smaller values
    # than the signal gradient)
    # In any way, gradient mean is only used as normalization here and using the mean is
    # alright (also image generation works normally).

    #print(src.data.std() * scale)
    if norm is not None and norm > 0.0:
        data_idx = batch_std(src.data) + _eps > norm / scale
        src.data[data_idx] =  (src.data / (batch_std(src.data, keepdim=True) + _eps) * norm / scale)[data_idx]

    if jitter > 0:
        # undo the shift
        src.data = roll(roll(src.data, -ox, -1), -oy, -2)

    if clip:
        src.data = torch.clamp(src.data, -bias / scale, (255 - bias) / scale)

    if blur:
        blur_in_place(src.data, sigma)


def deepdraw(net, base_img, octaves, random_crop=True, original_size=None,
             bias=None, scale=None, device='cuda', **step_params):
    """ Generate an image by iteratively optimizing activity of net.

    Arguments:
        net (nn.Module or function): A backpropagatable function/module that receives
            images in (B x C x H x W) form and outputs a scalar value per image.
        base_img (np.array): Initial image (h x w x c)
        octaves (list of dict): Configurations for each octave:
            n_iter (int): Number of iterations in this octave
            start_sigma (float): Initial standard deviation for gaussian smoothing (if
                used, see blur)
            end_sigma (float): Final standard deviation for gaussian smoothing (if used,
                see blur)
            start_step_size (float): Initial value of the step size used each iteration to
                update the image (im_old += step_size * grad).
            end_step_size (float): Initial value of the step size used each iteration to
                update the image (im_old += step_size * grad).
            (optionally) scale (float): If set, the image will be scaled using this factor
                during optimization. (Original image size is left unchanged).
        random_crop (boolean): If image to optimize is bigger than networks input image,
            optimize random crops of the image each iteration.
        original_size (triplet): (channel, height, width) expected by the network. If
            None, it uses base_img's.
        bias (float), scale (float): Values used for image normalization (at the very
            start of processing): (base_img - bias) / scale.
        device (torch.device or str): Device where the network is located.
        step_params (dict): A handful of optional parameters that are directly sent to
            make_step() (see docstring of make_step for a description).

    Returns:
        A h x w array. The optimized image.
    """
    # prepare base image
    image = process(base_img, mu=bias, sigma=scale)  # (3,224,224)

    # get input dimensions from net
    if original_size is None:
        print('getting image size:')
        c, w, h = image.shape[-3:]
    else:
        c, w, h = original_size

    print("starting drawing")

    src = torch.zeros(1, c, w, h, requires_grad=True, device=device)

    for e, o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = ndimage.zoom(image, (1, o['scale'], o['scale']))
        _, imw, imh = image.shape
        for i in range(o['iter_n']):
            if imw > w:
                if random_crop:
                    # randomly select a crop
                    # ox = random.randint(0,imw-224)
                    # oy = random.randint(0,imh-224)
                    mid_x = (imw - w) / 2.
                    width_x = imw - w
                    ox = np.random.normal(mid_x, width_x * 0.3, 1)
                    ox = int(np.clip(ox, 0, imw - w))
                    mid_y = (imh - h) / 2.
                    width_y = imh - h
                    oy = np.random.normal(mid_y, width_y * 0.3, 1)
                    oy = int(np.clip(oy, 0, imh - h))
                    # insert the crop into src.data[0]
                    src.data[0].copy_(torch.Tensor(image[:, ox:ox + w, oy:oy + h]))
                else:
                    ox = int((imw - w) / 2)
                    oy = int((imh - h) / 2)
                    src.data[0].copy_(torch.Tensor(image[:, ox:ox + w, oy:oy + h]))
            else:
                ox = 0
                oy = 0
                src.data[0].copy_(torch.Tensor(image))

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

            make_step(net, src, bias=bias, scale=scale, sigma=sigma, step_size=step_size, **step_params)

            if i % 10 == 0:
                print('finished step %d in octave %d' % (i, e))

            # insert modified image back into original image (if necessary)
            image[:, ox:ox + w, oy:oy + h] = src.data[0].cpu().numpy()

    # returning the resulting image
    return unprocess(image, mu=bias, sigma=scale)

octaves0 = [
 {
        'layer':'conv5',
        'iter_n':600,
        'start_sigma':1.5,
        'end_sigma':0.01,
        'start_step_size': 12.*0.25,
        'end_step_size':0.5*0.25,
    },
]


def diverse_deepdraw(net, base_img, octaves, random_crop=True, original_size=None,
                     bias=None, scale=None, device='cuda', div_metric='correlation',
                     div_linkage='minimum', div_weight=0, div_mask=1, **step_params):
    """ Similar to deepdraw() but including a diversity term among all images a la
    Cadena et al., 2018.

    Arguments (only those additional to deepdraw):
        base_img: (CHANGED) Expects a 4-d array (num_images x height x width x channels).
        div_metric (str): What metric to use when computing pairwise differences.
        div_linkage (str): How to agglomerate pairwise distances.
        div_weight (float): Weight given to the diversity term in the objective function.
        div_mask (np.array): Array (height x width) used to mask irrelevant parts of the
            image before calculating diversity.
    """
    if len(base_img) < 2:
        raise ValueError('You need to pass at least two initial images. Did you mean to '
                         'use deepdraw()?')

    # prepare base image
    image = process(base_img, mu=bias, sigma=scale)  # (num_batches, num_channels, h, w)

    # get input dimensions from net
    if original_size is None:
        print('getting image size:')
        c, w, h = image.shape[-3:]
    else:
        c, w, h = original_size

    print("starting drawing")

    src = torch.zeros(len(image), c, w, h, requires_grad=True, device=device)
    mask = torch.tensor(div_mask, dtype=torch.float32, device=device)

    for e, o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = ndimage.zoom(image, (1, 1, o['scale'], o['scale']))
        imw, imh = image.shape[-2:]
        for i in range(o['iter_n']):
            if imw > w:
                if random_crop:
                    # randomly select a crop
                    # ox = random.randint(0,imw-224)
                    # oy = random.randint(0,imh-224)
                    mid_x = (imw - w) / 2.
                    width_x = imw - w
                    ox = np.random.normal(mid_x, width_x * 0.3, 1)
                    ox = int(np.clip(ox, 0, imw - w))
                    mid_y = (imh - h) / 2.
                    width_y = imh - h
                    oy = np.random.normal(mid_y, width_y * 0.3, 1)
                    oy = int(np.clip(oy, 0, imh - h))
                    # insert the crop into src.data[0]
                    src.data[:].copy_(torch.Tensor(image[..., ox:ox + w, oy:oy + h]))
                else:
                    ox = int((imw - w) / 2)
                    oy = int((imh - h) / 2)
                    src.data[:].copy_(torch.Tensor(image[..., ox:ox + w, oy:oy + h]))
            else:
                ox = 0
                oy = 0
                src.data[:].copy_(torch.Tensor(image))

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

            div_term = 0
            if div_weight > 0:
                # Compute distance matrix
                images = (src * mask).view(len(src), -1)  # num_images x num_pixels
                if div_metric == 'correlation':
                    # computations restricted to the mask
                    means = (images.sum(dim=-1) / mask.sum()).view(len(images), 1, 1, 1)
                    residuals = ((src - means) * torch.sqrt(mask)).view(len(src), -1)
                    ssr = (((src - means) ** 2) * mask).sum(-1).sum(-1).sum(-1)
                    distance_matrix = -(torch.mm(residuals, residuals.t()) /
                                        torch.sqrt(torch.ger(ssr, ssr)) + 1e-12)
                elif div_metric == 'cosine':
                    image_norms = torch.norm(images, dim=-1)
                    distance_matrix = -(torch.mm(images, images.t()) /
                                        (torch.ger(image_norms, image_norms) + 1e-12))
                elif div_metric == 'euclidean':
                    distance_matrix = torch.norm(images.unsqueeze(0) -
                                                 images.unsqueeze(1), dim=-1)
                else:
                    raise ValueError('Invalid distance metric {} for the diversity term'.format(div_metric))

                # Compute overall distance in this image set
                triu_idx = torch.triu(torch.ones(len(distance_matrix),
                                                 len(distance_matrix)), diagonal=1) == 1
                if div_linkage == 'minimum':
                    distance = distance_matrix[triu_idx].min()
                elif div_linkage == 'average':
                    distance = distance_matrix[triu_idx].mean()
                else:
                    raise ValueError('Invalid linkage for the diversity term: {}'.format(div_linkage))

                div_term = div_weight * distance

            make_step(net, src, bias=bias, scale=scale, sigma=sigma, step_size=step_size,
                      add_loss=div_term, **step_params)

            # TODO: Maybe save the MEIs every number of iterations and return all MEIs.
            if i % 10 == 0:
                print('finished step %d in octave %d' % (i, e))

            # insert modified image back into original image (if necessary)
            image[..., ox:ox + w, oy:oy + h] = src.detach().cpu().numpy()

    # returning the resulting image
    return unprocess(image, mu=bias, sigma=scale)


def create_gabor(height=36, width=64, phase=0, wavelength=10, orientation=0, sigma=5,
                 dy=0, dx=0):
    """ Create a gabor patch (sinusoidal + gaussian).

    Arguments:
        height (int): Height of the image in pixels.
        width (int): Width of the image in pixels.
        phase (float): Angle at which to start the sinusoid in degrees.
        wavelength (float): Wavelength of the sinusoid (1 / spatial frequency) in pixels.
        orientation (float): Counterclockwise rotation to apply (0 is horizontal) in
            degrees.
        sigma (float): Sigma of the gaussian mask used in pixels
        dy (float): Amount of translation in y (positive moves down) in pixels/height.
        dx (float): Amount of translation in x (positive moves right) in pixels/height.

    Returns:
        Array of height x width shape with the required gabor.
    """
    # Compute image size to avoid translation or rotation producing black spaces
    padding = max(height, width)
    imheight = height + 2 * padding
    imwidth = width + 2 * padding
    # we could have diff pad sizes per dimension = max(dim_size, sqrt((h/2)^2 + (w/2)^2))
    # but this simplifies the code for just a bit of inefficiency

    # Create sinusoid with right wavelength and phase
    start_sample = phase
    step_size = 360 / wavelength
    samples = start_sample + step_size * np.arange(imheight)
    samples = np.mod(samples, 360)  # in degrees
    rad_samples = samples * (np.pi / 180)  # radians
    sin = np.sin(rad_samples)

    # Create Gabor by stacking the sinusoid along the cols
    gabor = np.tile(sin, (imwidth, 1)).T

    # Rotate around center
    gabor = ndimage.rotate(gabor, orientation, reshape=False)

    # Apply gaussian mask
    gaussy = signal.gaussian(imheight, std=sigma)
    gaussx = signal.gaussian(imwidth, std=sigma)
    mask = np.outer(gaussy, gaussx)
    gabor = gabor * mask

    # Translate (this is only approximate but it should be good enough)
    if abs(dx) > 1 or abs(dy) > 1:
        raise ValueError('Please express translations as factors of the height/width,'
                         'i.e, a number in interval [-1, 1] ')
    dy = int(dy * height)  # int is the approximation
    dx = int(dx * width)
    gabor = gabor[padding - dy: -padding - dy, padding - dx: -padding - dx]

    if gabor.shape != (height, width):
        raise ValueError('Dimensions of gabor do not match desired dimensions.')

    return gabor.astype(np.float32)





def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def rename(rel, prefix='new_', exclude=[]):
    attrs = list(rel.heading.attributes.keys())
    original = [x for x in attrs if x not in exclude]
    keys = [k for k in exclude if k in attrs]
    name_map = {prefix+x: x for x in original}
    return rel.proj(*keys, **name_map)


def plot_images(df, prefixes, names=None, brain_area='V1', n_rows=15, order_by='pearson',
                panels=('normed_rf', 'normed_mei'), panel_names=('RF', 'MEI'), cmaps=('coolwarm', 'gray'),
                y_infos=('{prefix}test_corr', 'pearson'), save_path=None):
    if names is None:
        names = prefixes

    f = (df['brain_area'] == brain_area)
    area_data = df[f]
    area_data = area_data.sort_values(order_by, ascending=False)

    n_rows = min(n_rows, len(area_data))
    n_panels = len(panels)
    cols = len(prefixes) * n_panels;

    with sns.axes_style('white'):
        fig, axs = plt.subplots(n_rows, cols, figsize=(4 * cols, round(2 * n_cells)))
        st = fig.suptitle('MEIs on Shuffled {} dataset: {}'.format(brain_area, ', '.join(names)))
        [ax.set_xticks([]) for ax in axs.ravel()]
        [ax.set_yticks([]) for ax in axs.ravel()]

    for ax_row, (_, data_row), row_index in zip(axs, area_data.iterrows(), count()):
        for ax_group, prefix, name in zip(grouper(n_panels, ax_row), prefixes, names):
            for ax, panel, panel_name, y_info, cm in zip(ax_group, panels, panel_names, y_infos, cmaps):
                if row_index == 0:
                    ax.set_title('{}: {}'.format(panel_name, name))
                ax.imshow(data_row[prefix + panel].squeeze(), cmap=cm)
                if y_info is not None:
                    ax.set_ylabel('{:0.2f}%'.format(data_row[y_info.format(prefix=prefix)] * 100))

    fig.tight_layout()

    # shift subplots down:
    st.set_y(0.98)
    st.set_fontsize(20)
    fig.subplots_adjust(top=0.95)
    if path is not None:
        fig.savefig(save_path)


def gen_gif(images, output_path, duration=5, scale=1, adj_single=False):
    h, w = images[0].shape
    imgsize = (w * scale, h * scale)
    images = np.stack([cv2.resize(img, imgsize) for img in images])

    axis = (1, 2) if adj_single else None
    images = images - images.min(axis=axis, keepdims=True)
    images = images / images.max(axis=axis, keepdims=True) * 255
    images = images.astype('uint8')

    single_duration = duration / len(images)
    if not output_path.endswith('.gif'):
        output_path += '.gif'
    imageio.mimsave(output_path, images, duration=single_duration)


def rescale_images(images, low=0, high=1, together=True):
    axis = None if together else (1, 2)
    images = images - images.min(axis=axis, keepdims=True)
    images = images / images.max(axis=axis, keepdims=True) * (high - low) + low
    return images


def scale_imagesize(images, scale=(2, 2)):
    h, w = images[0].shape
    imgsize = (w * scale[1], h * scale[0])
    return np.stack([cv2.resize(img, imgsize) for img in images])


def tile_images(images, rows, cols, vpad=0, hpad=0, normalize=False, base=0):
    n_images = len(images)
    assert rows * cols >= n_images
    h, w = images[0].shape

    total_image = np.zeros((h + (h + vpad) * (rows - 1), w + (w + hpad) * (cols - 1))) + base
    loc = product(range(rows), range(cols))
    for img, (i, j) in zip(images, loc):
        if normalize:
            img = rescale_images(img)
        voffset, hoffset = (h + vpad) * i, (w + hpad) * j
        total_image[voffset:voffset + h, hoffset:hoffset + w] = img
    return total_image


def repeat_frame(images, frame_pos=0, rep=4):
    parts = []
    if frame_pos < 0:
        frame_pos = len(images) + frame_pos

    if frame_pos > 0:
        parts.append(images[:frame_pos])
    parts.append(np.tile(images[frame_pos], (rep, 1, 1)))
    if frame_pos < len(images) - 1:
        parts.append(images[frame_pos+1:])
    return np.concatenate(parts)


def add_text(image, text, pos, fontsize=1, color=(0, 0, 0)):
    image = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, text, pos, font, fontsize, color, 1, cv2.LINE_8)
    return image