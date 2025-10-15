from torch.utils.data import DataLoader
from torchvision.datasets import MNIST as MNISTtorch
from torch.utils.data import random_split

import pytorch_lightning as pl


class MNIST(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, *args, **kwargs):
        super().__init__()
        self.val_data = None
        self.train_data = None
        self.test_data = None
        self.data_dir = data_dir
        self.args = args
        self.kwargs = kwargs
        self.batch_size = batch_size

        self.data_kwargs = {
            "root": self.kwargs.pop("root", self.data_dir),
            "download": self.kwargs.pop("download", True),
            "transform": self.kwargs.pop("transform", None),
        }

        self.kwargs.pop("train", "")

    def prepare_data(self):
        MNISTtorch(**self.data_kwargs)

    def setup(self, stage: str):
        dataset = MNISTtorch(**self.data_kwargs, train=True)
        train_set_size = int(len(dataset) * 0.9)
        valid_set_size = len(dataset) - train_set_size
        self.train_data, self.val_data = random_split(dataset, [train_set_size, valid_set_size])
        self.test_data = MNISTtorch(**self.data_kwargs, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, *self.args, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, *self.args, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size, *self.args, **self.kwargs)

    def dummy_input(self):
        return self.test_data[0][0].unsqueeze(0)
