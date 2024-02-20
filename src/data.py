#!/usr/bin/env python3

from glob import glob
import torchvision
import torch

from src import const


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.files = glob((const.DATA_DIR / '**' / '*.jpeg').as_posix(), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        X = torchvision.transforms.functional.resize(torchvision.io.read_image(self.files[idx]), const.IMAGE_SIZE, antialias=True)
        X = X / 255

        y = torch.zeros(len(const.CLASSES))
        y[const.CLASSES.index(self.files[idx].split('/')[-2])] = 1

        return [t.to(const.DEVICE) for t in (X, y)]
