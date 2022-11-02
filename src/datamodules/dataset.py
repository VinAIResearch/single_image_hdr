import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DrTMODataset(Dataset):
    rgb_to_y = torch.tensor([0.299, 0.587, 0.114]).view(-1, 1, 1)

    def __init__(
        self,
        labels_path="labels.csv",
        data_dir=".",
        transform=None,
        tau=0.95,
        test_hdr=False,
    ):
        super(DrTMODataset, self).__init__()

        labels = pd.read_csv(labels_path)
        labels["x"] = labels["x"].apply(lambda path: os.path.join(data_dir, path))
        labels["y"] = labels["y"].apply(lambda path: os.path.join(data_dir, path))
        labels["x_exp y_exp".split()] = labels["x_exp y_exp".split()].astype(np.float32)

        self.labels = labels
        self.transform = transform
        self.tau = tau
        self.test_hdr = test_hdr

    def __getitem__(self, index):
        item = self.labels.iloc[index]

        x_name = item["x"]
        y_name = item["y"]
        x = self.load_img(x_name)
        # Just a trick for testing single-image hdr datasets cause
        # I don't have enough time to code it clean
        y = self.load_img(y_name) if not self.test_hdr else x
        if self.transform:
            transformed = self.transform(image=x, gt=y)
            x = transformed["image"]
            y = transformed["gt"]

        x_exp = item["x_exp"]
        y_exp = item["y_exp"]

        if x_exp > y_exp:
            x_name, y_name = y_name, x_name
            x, y = y, x
            x_exp, y_exp = y_exp, x_exp

        x_mask, y_mask = self._get_masks_(x, y)

        return {
            "x": x,
            "y": y,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "x_exp": x_exp,
            "y_exp": y_exp,
            "x_name": x_name,
            "y_name": y_name,
        }

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_img(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _get_well_exposed_masks_(self, img):
        # Take mask on Y channel only
        Y = (img * self.rgb_to_y).sum(dim=0, keepdim=True)

        under_exposed_mask = (1 - self.tau - Y).clip(min=0.0) / (1 - self.tau)
        over_exposed_mask = (Y - self.tau).clip(min=0.0) / (1 - self.tau)

        return 1 - torch.max(under_exposed_mask, over_exposed_mask)

    def _get_masks_(self, I1, I2):
        return self._get_well_exposed_masks_(I1), self._get_well_exposed_masks_(I2)
