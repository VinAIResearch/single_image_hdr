import math
from functools import partial

import torch
import torch.nn as nn

from .act_norm import act, norm
from .arch_utils import enable_bias


def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
    """Fills the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        inizializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = inizializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel


class SubPixelUpsampling(nn.Module):
    def __init__(
        self,
        ins,
        ks=1,
        ps=0,
        stride=1,
        norm_type="syncbatch",
        act_type="relu",
        *args,
        **kwargs
    ):
        super().__init__()

        self.out_channels = ins
        self.conv_shuffle = nn.Conv2d(
            ins,
            ins * 4,
            kernel_size=ks,
            stride=stride,
            padding=ps,
            bias=enable_bias(norm_type),
        )
        self.abn = nn.Sequential(
            norm(norm_type, ins * 4), act(act_type), nn.PixelShuffle(upscale_factor=2),
        )

        self.conv_shuffle.weight.data.copy_(
            ICNR(
                self.conv_shuffle.weight,
                upscale_factor=2,
                inizializer=partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
            )
        )
        if self.conv_shuffle.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_shuffle.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv_shuffle.bias, -bound, bound)

    def forward(self, x):
        x = self.conv_shuffle(x)
        x = self.abn(x)
        return x
