from argparse import ArgumentParser
from functools import partial
from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from src.architectures.base_unet import BaseUNet
from src.architectures.modules import (
    SubPixelUpsampling,
    UNetDecBlock,
    UNetEncBlock,
    weights_init,
)

from .losses import LossFunction


class EDNet(LightningModule):
    def __init__(self, hparams):
        super(EDNet, self).__init__()

        self.save_hyperparameters(hparams)

        # self.example_input_array = (torch.rand(1, 3, 256, 256), torch.tensor([0]))

        baseEncBlk = partial(
            UNetEncBlock,
            norm_type=self.hparams.norm,
            act_type=self.hparams.act_enc,
            learn_shortcut=self.hparams.shortcut,
        )
        baseDecBlk = partial(
            UNetDecBlock,
            norm_type=self.hparams.norm,
            act_type=self.hparams.act_dec,
            learn_shortcut=self.hparams.shortcut,
        )
        baseDownsampler = partial(nn.MaxPool2d, kernel_size=2, stride=2)
        baseUpsampler = partial(
            SubPixelUpsampling,
            norm_type=self.hparams.norm,
            act_type=self.hparams.act_dec,
        )
        baseLastConvLayer = partial(nn.Conv2d, kernel_size=1)
        BaseSubNet = partial(
            BaseUNet,
            enc_blk=baseEncBlk,
            dec_blk=baseDecBlk,
            downsampler=baseDownsampler,
            upsampler=baseUpsampler,
            in_channels=3,
            out_channels=3,
        )

        self.HDREncNet = BaseSubNet(
            nfeats=self.hparams.n1_filters,
            maxfeats=self.hparams.n1_maxfeats,
            nlayers=self.hparams.n1_layers,
            global_res=self.hparams.n1_global_res,
            last_conv={"conv": baseLastConvLayer, "act_type": "tanh"},
        )
        self.UpExposureNet = BaseSubNet(
            nfeats=self.hparams.n23_filters,
            maxfeats=self.hparams.n23_maxfeats,
            nlayers=self.hparams.n23_layers,
            global_res=self.hparams.n23_global_res,
            last_conv={"conv": baseLastConvLayer, "act_type": "normalizedtanh"},
        )
        self.DownExposureNet = BaseSubNet(
            nfeats=self.hparams.n23_filters,
            maxfeats=self.hparams.n23_maxfeats,
            nlayers=self.hparams.n23_layers,
            global_res=self.hparams.n23_global_res,
            last_conv={"conv": baseLastConvLayer, "act_type": "normalizedtanh"},
        )

        if self.hparams.weights_init:
            self.HDREncNet.apply(weights_init)
            self.UpExposureNet.apply(weights_init)
            self.DownExposureNet.apply(weights_init)

        self.loss_fn = LossFunction(
            weights={"hdr": 1e0, "recon": 1e2, "TV": 1e-5, "content": 1e-1}
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Training phrase
        # ---------------
        # Model
        parser.add_argument("--norm", type=str, default="batch")
        parser.add_argument("--act_enc", type=str, default="mish")
        parser.add_argument("--act_dec", type=str, default="mish")
        parser.add_argument("--shortcut", action="store_true")
        parser.add_argument("--weights_init", action="store_true")
        parser.add_argument("--n1_filters", type=int, default=16)
        parser.add_argument("--n1_maxfeats", type=int, default=256)
        parser.add_argument("--n1_layers", type=int, default=7)
        parser.add_argument("--n1_global_res", type=str, default="add")
        parser.add_argument("--n23_filters", type=int, default=32)
        parser.add_argument("--n23_maxfeats", type=int, default=512)
        parser.add_argument("--n23_layers", type=int, default=7)
        parser.add_argument("--n23_global_res", type=str, default="concat")

        # Optimizer
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        # Logging
        parser.add_argument("--multigpu", action="store_true")

        return parser

    @staticmethod
    def cal_psnr(x, y):
        # Constant for numerical stability
        EPS = 1e-8

        mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
        score = -10 * torch.log10(mse + EPS)

        return score.mean(dim=0)

    def forward(self, I1, mask, t2, isUpExposed=False):
        E_tilde, _ = self.HDREncNet(I1, mask=mask)
        E_hat = (I1 + E_tilde).relu()
        if isUpExposed:
            I2_hat, _ = self.UpExposureNet(E_hat * t2)
        else:
            I2_hat, _ = self.DownExposureNet(E_hat * t2)
        return I2_hat, E_hat

    def step(self, batch):
        x = batch["x"]
        y = batch["y"]
        x_mask = batch["x_mask"]
        y_mask = batch["y_mask"]
        x_exp = batch["x_exp"].view(-1, 1, 1, 1)
        y_exp = batch["y_exp"].view(-1, 1, 1, 1)
        x_wrt_y_exp = x_exp / y_exp
        y_wrt_x_exp = y_exp / x_exp

        y_hat, E1 = self(x, x_mask, y_wrt_x_exp, isUpExposed=True)
        x_hat, E2 = self(y, y_mask, x_wrt_y_exp, isUpExposed=False)

        # Loss calculation
        losses = self.loss_fn(
            E1=E1,
            E1_hat=E2 * x_wrt_y_exp,
            E2=E2,
            E2_hat=E1 * y_wrt_x_exp,
            x=x,
            x_hat=x_hat,
            y=y,
            y_hat=y_hat,
            x_mask=x_mask,
            y_mask=y_mask,
        )

        return losses, x, x_hat, y, y_hat

    def get_losses_log(self, losses, state="train"):
        return {
            f"{state}/loss": losses["total"],
            f"{state}/up_hdr_loss": losses["up_hdr"],
            f"{state}/up_recon_loss": losses["up_recon"],
            f"{state}/up_TV_loss": losses["up_TV"],
            f"{state}/up_content_loss": losses["up_content"],
            f"{state}/down_hdr_loss": losses["down_hdr"],
            f"{state}/down_recon_loss": losses["down_recon"],
            f"{state}/down_TV_loss": losses["down_TV"],
            f"{state}/down_content_loss": losses["down_content"],
        }

    def get_metrics_log(self, x_hat, x, y_hat, y, state="train"):
        # Metrics calculation
        up_psnr = self.cal_psnr(y_hat, y)
        down_psnr = self.cal_psnr(x_hat, x)
        psnr = 0.5 * (up_psnr + down_psnr)

        non_display_metrics_log = {
            f"{state}/up_psnr": up_psnr,
            f"{state}/down_psnr": down_psnr,
        }
        display_metrics_log = {
            f"{state}/psnr": psnr,
        }
        return non_display_metrics_log, display_metrics_log

    def training_step(self, batch, batch_idx):
        losses, x, x_hat, y, y_hat = self.step(batch)

        losses_log = self.get_losses_log(losses, "train")
        non_display_metrics_log, display_metrics_log = self.get_metrics_log(
            x_hat, x, y_hat, y, "train"
        )
        peak_mem_usage = {
            "mem": (
                getrusage(RUSAGE_SELF).ru_maxrss + getrusage(RUSAGE_CHILDREN).ru_maxrss
            )
            / 1048576
        }  # As Gb, tested on linux only

        self.log_dict(
            peak_mem_usage, prog_bar=True, on_step=True, on_epoch=False,
        )
        self.log_dict(
            losses_log, prog_bar=False, on_step=False, on_epoch=True,
        )
        self.log_dict(
            non_display_metrics_log, prog_bar=False, on_step=True, on_epoch=False,
        )
        self.log_dict(
            display_metrics_log, prog_bar=True, on_step=True, on_epoch=False,
        )

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        losses, x, x_hat, y, y_hat = self.step(batch)

        losses_log = self.get_losses_log(losses, "val")
        non_display_metrics_log, display_metrics_log = self.get_metrics_log(
            x_hat, x, y_hat, y, "val"
        )

        self.log(
            "val/loss",
            losses_log.pop("val/loss"),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.hparams.multigpu,
        )
        self.log_dict(
            losses_log,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=self.hparams.multigpu,
        )
        self.log_dict(
            non_display_metrics_log,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=self.hparams.multigpu,
        )
        self.log_dict(
            display_metrics_log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.hparams.multigpu,
        )

    def test_step(self, batch, batch_idx):
        losses, x, x_hat, y, y_hat = self.step(batch)

        non_display_metrics_log, display_metrics_log = self.get_metrics_log(
            x_hat, x, y_hat, y, "test"
        )

        self.log(
            "test/loss",
            losses["total"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.hparams.multigpu,
        )
        self.log_dict(
            non_display_metrics_log,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=self.hparams.multigpu,
        )
        self.log_dict(
            display_metrics_log,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.hparams.multigpu,
        )

        return losses["total"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, verbose=True, patience=4,
        )

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "reduce_on_plateau": True,
            "monitor": "val/loss",
        }
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
