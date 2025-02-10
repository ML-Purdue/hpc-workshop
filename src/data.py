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

        return X, y

# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
