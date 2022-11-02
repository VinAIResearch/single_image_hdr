import torch.nn as nn

from .act_norm import act, norm
from .arch_utils import enable_bias


class UNetEncBlock(nn.Module):
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


class UNetDilatedEncBlock(nn.Module):
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
