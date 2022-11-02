from functools import partial

import torch.nn as nn


def normalized_tanh(x, inplace: bool = False):
    return 0.5 * x.tanh() + 0.5


class NormalizedTanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super(NormalizedTanh, self).__init__()

    def forward(self, x):
        return normalized_tanh(x)


ACTS = {
    "relu": nn.ReLU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=2e-1),
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "normalizedtanh": NormalizedTanh,
    "mish": nn.Mish,
    "identity": nn.Identity,
}


NORMS = {
    "instance": nn.InstanceNorm2d,
    "batch": nn.BatchNorm2d,
    "syncbatch": nn.SyncBatchNorm,
    "layer": partial(nn.GroupNorm, 1),
    "identity": nn.Identity,
}


def act(name="relu", *args, **kwargs):
    return ACTS[name](*args, **kwargs)


def norm(name="batch", *args, **kwargs):
    return NORMS[name](*args, **kwargs)
