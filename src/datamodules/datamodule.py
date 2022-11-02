from argparse import ArgumentParser

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import DrTMODataset


class DrTMODataModule(LightningDataModule):
    def __init__(
        self,
        num_workers=0,
        batch_size=16,
        test_batch_size=1,
        pin_memory=False,
        image_size=512,
        clip_threshold=0.95,
        train_dir=None,
        test_dir=None,
        train_label_path=None,
        val_label_path=None,
        test_label_path=None,
        test_hdr=False,
        *args,
        **kwargs
    ):
        super(*args, **kwargs).__init__()

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.clip_threshold = clip_threshold

        self.train_dir = train_dir
        self.train_label_path = train_label_path
        self.val_label_path = val_label_path
        self.test_dir = test_dir
        self.test_label_path = test_label_path
        self.test_hdr = test_hdr

        # self.dims is returned when you call datamodule.size()
        self.dims = (self.batch_size, 3, self.image_size, self.image_size)

        self.train_transforms = A.Compose(
            [
                A.ToFloat(max_value=255.0),
                A.RandomCrop(self.image_size, self.image_size),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.0,
                    rotate_limit=179,
                    interpolation=cv2.INTER_CUBIC,
                ),
                ToTensorV2(),
            ],
            additional_targets={"gt": "image"},
        )
        self.test_transforms = self.val_transforms = A.Compose(
            [A.ToFloat(max_value=255.0), ToTensorV2()],
            additional_targets={"gt": "image"},
        )

        self.data_train = None
        self.data_val = None
        self.data_test = None

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Dataset
        parser.add_argument("--train_dir", type=str, default="./train")
        parser.add_argument("--train_label_path", type=str, default="./train.csv")
        parser.add_argument("--val_label_path", type=str, default="./val.csv")
        parser.add_argument("--test_dir", type=str, default="./test")
        parser.add_argument("--test_label_path", type=str, default="./test.csv")
        parser.add_argument("--test_hdr", action="store_true")

        # DataLoader
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--pin_memory", action="store_true")
        parser.add_argument("--image_size", type=int, default=256)

        parser.add_argument("--clip_threshold", type=float, default=0.95)

        return parser

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.train_dir is not None:
            self.data_train = DrTMODataset(
                self.train_label_path,
                self.train_dir,
                self.train_transforms,
                tau=self.clip_threshold,
                test_hdr=False,
            )
            self.data_val = DrTMODataset(
                self.val_label_path,
                self.train_dir,
                self.val_transforms,
                tau=self.clip_threshold,
                test_hdr=False,
            )
        if self.test_dir is not None:
            self.data_test = DrTMODataset(
                self.test_label_path,
                self.test_dir,
                self.test_transforms,
                tau=self.clip_threshold,
                test_hdr=self.test_hdr,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
