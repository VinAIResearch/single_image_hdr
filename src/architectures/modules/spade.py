import torch.nn as nn
import torch.nn.functional as F

from .act_norm import act, norm


class SPADE(nn.Module):
    def __init__(
        self, norm_type, norm_nc, mask_nc, nhidden=128, act_type="relu", *args, **kwargs
    ):
        super().__init__()

        self.param_free_norm = norm(norm_type, norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(mask_nc, nhidden, kernel_size=3, padding=1), act(act_type)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, mask):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on the input mask
        mask = F.interpolate(mask, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(mask)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out
