import piq
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two or more loss functions into one.
    This class computes a weighted sum of all losses.
    """

    def __init__(self, losses, weights):
        super().__init__()
        self.losses = [
            WeightedLoss(loss, weight) for loss, weight in zip(losses, weights)
        ]

    def forward(self, *input):
        total_loss = 0
        for loss_fn in self.losses:
            total_loss += loss_fn(*input)
        return total_loss


def total_variation(
    x: torch.Tensor, reduction: str = "mean", norm_type: str = "l2"
) -> torch.Tensor:
    r"""Compute Total Variation metric
    Source: https://github.com/photosynthesis-team/piq/blob/60abd2f22eafa2c8af57006525e8dff4dbbb43eb/piq/tv.py

    Args:
        x: Tensor with shape (N, C, H, W).
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        norm_type: {'l1', 'l2', 'l2_squared'}, defines which type of norm to implement, isotropic or anisotropic.

    Returns:
        score : Total variation of a given tensor

    References:
        https://www.wikiwand.com/en/Total_variation_denoising
        https://remi.flamary.com/demos/proxtv.html
    """

    norm_fn = {"l1": torch.abs, "l2": torch.pow, "l2_squared": torch.pow}[norm_type]

    w_variance = torch.sum(norm_fn(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=[1, 2, 3])
    h_variance = torch.sum(norm_fn(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=[1, 2, 3])

    if norm_type == "l2":
        score = torch.sqrt(h_variance + w_variance)
    else:
        score = h_variance + w_variance

    if reduction == "none":
        return score

    return {"mean": score.mean, "sum": score.sum}[reduction](dim=0)


class TVLoss(_Loss):
    r"""Creates a criterion that measures the total variation of the
    the given input :math:`x`.
    Source: https://github.com/photosynthesis-team/piq/blob/60abd2f22eafa2c8af57006525e8dff4dbbb43eb/piq/tv.py


    If :attr:`norm_type` set to ``'l2'`` the loss can be described as:

    .. math::
        TV(x) = \sum_{N}\sqrt{\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}|^2 +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|^2)}

    Else if :attr:`norm_type` set to ``'l1'``:

    .. math::
        TV(x) = \sum_{N}\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}| +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|) $$

    where :math:`N` is the batch size, `C` is the channel size.

    Args:
        norm_type: one of {'l1', 'l2', 'l2_squared'}
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Shape:
        - Input: Required to be 2D (H, W), 3D (C,H,W) or 4D (N,C,H,W)
    Examples::

        >>> loss = TVLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> output = loss(prediction)
        >>> output.backward()

    References:
        https://www.wikiwand.com/en/Total_variation_denoising
        https://remi.flamary.com/demos/proxtv.html
    """

    def __init__(self, norm_type: str = "l2", reduction: str = "mean"):
        super().__init__()

        self.norm_type = norm_type
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        r"""Computation of Total Variation (TV) index as a loss function.

        Args:
            prediction: Tensor of prediction of the network.

        Returns:
            Value of TV loss to be minimized.
        """
        score = total_variation(
            prediction, reduction=self.reduction, norm_type=self.norm_type,
        )
        return score


class LossFunction(_Loss):
    def __init__(
        self, weights={"hdr": 1e0, "recon": 1e2, "TV": 5e-4, "content": 1e-2}
    ):
        super().__init__()

        self.hdr_loss = nn.L1Loss()
        self.recon_loss = nn.L1Loss(reduction="none")
        self.TV_loss = TVLoss(norm_type="l1")
        self.content_loss = piq.ContentLoss(
            feature_extractor="vgg19",
            layers=("pool1", "pool2", "pool3"),
            weights=[1.0] * 3,
            replace_pooling=False,
            distance="mae",
            normalize_features=False,
        )

        self.register_buffer("hdr_weight", torch.tensor(weights["hdr"]))
        self.register_buffer("recon_weight", torch.tensor(weights["recon"]))
        self.register_buffer("TV_weight", torch.tensor(weights["TV"]))
        self.register_buffer("content_weight", torch.tensor(weights["content"]))

    def _loss_preprocess(self, E, epsilon=1.0 / 255.0):
        return torch.log(E + epsilon)

    def forward(
        self, E1, E1_hat, E2, E2_hat, x, x_hat, y, y_hat, y_mask=None, x_mask=None
    ):
        up_hdr_loss = self.hdr_loss(
            self._loss_preprocess(E2_hat), self._loss_preprocess(E2)
        )
        up_recon_loss = (self.recon_loss(y_hat, y) * y_mask).mean()
        up_TV_loss = self.TV_loss(y_hat)
        up_content_loss = self.content_loss(y_hat * y_mask, y * y_mask)

        down_hdr_loss = self.hdr_loss(
            self._loss_preprocess(E1_hat), self._loss_preprocess(E1)
        )
        down_recon_loss = (self.recon_loss(x_hat, x) * x_mask).mean()
        down_TV_loss = self.TV_loss(x_hat)
        down_content_loss = self.content_loss(x_hat * x_mask, x * x_mask)

        loss = (
            self.recon_weight * (up_recon_loss + down_recon_loss)
            + self.hdr_weight * (up_hdr_loss + down_hdr_loss)
            + self.TV_weight * (up_TV_loss + down_TV_loss)
            + self.content_weight * (up_content_loss + down_content_loss)
        )

        return {
            "total": loss,
            "up_hdr": up_hdr_loss,
            "up_recon": up_recon_loss,
            "up_TV": up_TV_loss,
            "up_content": up_content_loss,
            "down_hdr": down_hdr_loss,
            "down_recon": down_recon_loss,
            "down_TV": down_TV_loss,
            "down_content": down_content_loss,
        }
