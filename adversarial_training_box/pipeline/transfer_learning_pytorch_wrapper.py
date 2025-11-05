import torch
from torch import nn
from typing import Any, Iterator

import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import Accuracy


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, name, lr=0.02, loss=F.cross_entropy, k=1, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model: torch.nn.Module = model
        self.lr = lr
        self.loss = loss
        self.name = name
        self.last_weights = None
        self.k = k

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        y_hat = self.model(x)
        # print(y_hat)
        # print(y)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        acc = torch.sum(torch.argmax(y_hat, dim=-1) == y) / torch.numel(y)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        self.model.train(False)
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        # TODO: make variable
        accuracy = Accuracy("multiclass", num_classes=47, top_k=self.k)
        acc = accuracy(y_hat, y)
        self.log('accuracy', acc, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.train(False)
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def only_train_k_layers(self, k):
        """Only activate training for the last k layers"""
         # First, enable gradients for all parameters
        self.model.requires_grad_(True)
        
        if k < 0:  # train all layers
            return
        
        # Get all children as a list
        children_list = list(self.model.children())
        
        # Freeze all layers except the last k
        layers_to_freeze = children_list[:-k] if k < len(children_list) else []
        
        for layer in layers_to_freeze:
            # Freeze all parameters in this layer
            for param in layer.parameters():
                param.requires_grad = False

    def reset_k_layers(self, k):
        """Reset the weights of the last k layers

        Return:
            True if the last layer had parameters to reset, False otherwise
        """
        could_reset = False
        # go through each child of the model until child k, backwards
        for layer in list(self.model.children())[:-k-1:-1]:
            # TODO: possible logic error
            could_reset = False
            if hasattr(layer, 'reset_parameters'):
                could_reset = True
                # TODO uncomment
                layer.reset_parameters()
        return could_reset

    def children(self) -> Iterator['Module']:
        return self.model.children()

    # def on_after_backward(self) -> None:
        # if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
        # if self.last_weights is not None:
        #     print(self.last_weights)
        #     print(self.last_weights - torch.Tensor(list(list(self.model.children())[-1].parameters())[0].clone()))
        # self.last_weights = torch.Tensor(list(list(self.model.children())[-1].parameters())[0].clone())
