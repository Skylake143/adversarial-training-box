from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST as EMNISTtorch
from torch.utils.data import random_split

import pytorch_lightning as pl


class EMNIST(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, split="balanced", *args, **kwargs):
        super().__init__()
        self.val_data = None
        self.train_data = None
        self.test_data = None
        self.data_dir = data_dir
        self.args = args
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.split = split

        self.data_kwargs = {
            "root": self.kwargs.pop("root", self.data_dir),
            "download": self.kwargs.pop("download", True),
            "transform": self.kwargs.pop("transform", None),
        }

        self.kwargs.pop("train", "")

    def prepare_data(self):
        EMNISTtorch(**self.data_kwargs, split=self.split)

    def setup(self, stage: str):
        dataset = EMNISTtorch(**self.data_kwargs, train=True, split=self.split)
        train_set_size = int(len(dataset) * 0.9)
        valid_set_size = len(dataset) - train_set_size
        self.train_data, self.val_data = random_split(dataset, [train_set_size, valid_set_size])
        self.test_data = EMNISTtorch(**self.data_kwargs, train=False, split=self.split)

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, *self.args, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, *self.args, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size, *self.args, **self.kwargs)

    def dummy_input(self):
        return self.test_data[0][0].unsqueeze(0)
