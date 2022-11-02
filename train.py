import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, plugins, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

from src.datamodules.datamodule import DrTMODataModule
from src.models.ednet import EDNet


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


def main(args, ckptCallback):
    args.seed = seed_everything(args.seed)

    if args.accelerator == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus * args.num_nodes))
        args.num_workers = int(args.num_workers / max(1, args.gpus * args.num_nodes))
        ddp_plugin = plugins.DDPPlugin(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            ddp_comm_hook=default.fp16_compress_hook if args.fp16_hook else None,
        )
    else:
        ddp_plugin = None

    dataset = DrTMODataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        image_size=args.image_size,
        clip_threshold=args.clip_threshold,
        train_dir=args.train_dir,
        train_label_path=args.train_label_path,
        val_label_path=args.val_label_path,
    )

    model = EDNet(hparams=args)

    try:
        args.ver = int(args.ver)
    except ValueError:
        pass
    logger = TensorBoardLogger("logs", name=args.ckpt_filename, version=args.ver)

    set_debug_apis(state=False)
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=[ckptCallback], plugins=ddp_plugin
    )
    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add reproducibility args
    parser.add_argument("--seed", default=131898, type=int)

    # Add callbacks args
    parser.add_argument("--ckpt_filename", default="default", type=str)
    parser.add_argument("--ckpt_dir", default="ckpts/", type=str)
    parser.add_argument("--ckpt_period", default=1, type=int)

    # Add logger args
    parser.add_argument("--ver", default=0)

    # Add ddp optmization args
    parser.add_argument("--fp16_hook", default=True, type=bool)

    # Add model specific args
    parser = EDNet.add_model_specific_args(parser)

    # Add dataset specific args
    parser = DrTMODataModule.add_dataset_specific_args(parser)

    # Add trainer args
    # Need to specify addition options when training, like
    # --benchmark=True --precision=16
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    args.ckpt_dir = os.path.join(args.ckpt_dir, args.ckpt_filename)
    try:
        os.mkdir(args.ckpt_dir)
    except FileExistsError:
        pass

    ckptCallback = ModelCheckpoint(
        monitor="val/loss",
        save_last=True,
        save_top_k=16,
        every_n_epochs=args.ckpt_period,
        mode="min",
        dirpath=args.ckpt_dir,
        filename=args.ckpt_filename
        + "-{epoch}-val-loss{val/loss:.4f}-psnr{val/psnr:.4f}",
        auto_insert_metric_name=False,
    )

    # training
    main(args, ckptCallback)
