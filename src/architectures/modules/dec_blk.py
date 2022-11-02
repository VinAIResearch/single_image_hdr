import torch.nn as nn

from .act_norm import act, norm
from .arch_utils import enable_bias
from .spade import SPADE


class UNetDecBlock(nn.Module):
    def __init__(
        self,
        ins,
        outs,
        norm_type="syncbatch",
        act_type="relu",
        learn_shortcut=False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.learn_shortcut = learn_shortcut

        self.blk = nn.Sequential(
            nn.Conv2d(ins, outs, 3, 1, 1, bias=enable_bias(norm_type)),
            norm(norm_type, outs),
            act(act_type),
            nn.Conv2d(outs, outs, 3, 1, 1, bias=enable_bias(norm_type)),
            norm(norm_type, outs),
            act(act_type),
        )
        if learn_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ins, outs, 1, 1, 0, bias=enable_bias(norm_type)),
                norm(norm_type, outs),
                act(act_type),
            )

    def forward(self, x):
        if self.learn_shortcut:
            return self.blk(x) + self.shortcut(x)
        else:
            return self.blk(x)


class UNetDilatedDecBlock(nn.Module):
    def __init__(
        self,
        ins,
        outs,
        norm_type="syncbatch",
        act_type="relu",
        learn_shortcut=False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.learn_shortcut = learn_shortcut

        self.blk = nn.Sequential(
            nn.Conv2d(ins, outs, 3, 1, 2, 2, bias=enable_bias(norm_type)),
            norm(norm_type, outs),
            act(act_type),
            nn.Conv2d(outs, outs, 3, 1, 2, 2, bias=enable_bias(norm_type)),
            norm(norm_type, outs),
            act(act_type),
        )
        if learn_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ins, outs, 3, 1, 2, 2, bias=enable_bias(norm_type)),
                norm(norm_type, outs),
                act(act_type),
            )

    def forward(self, x):
        if self.learn_shortcut:
            return self.blk(x) + self.shortcut(x)
        else:
            return self.blk(x)


class UNetDecSPADEBlock(nn.Module):
    def __init__(
        self, ins, outs, norm_type="syncbatch", act_type="relu", *args, **kwargs
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(ins, outs, 3, 1, 1, 1, bias=False)
        self.norm1 = SPADE(norm_type, outs, 1, act_type=act_type)
        self.act1 = act(act_type)
        self.conv2 = nn.Conv2d(outs, outs, 3, 1, 1, 1, bias=False)
        self.norm2 = SPADE(norm_type, outs, 1, act_type=act_type)
        self.act2 = act(act_type)

    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.norm1(x, mask)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x, mask)
        x = self.act2(x)
        return x
