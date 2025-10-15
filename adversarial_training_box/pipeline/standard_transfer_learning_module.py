from email.mime import base
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from pytorch_lightning.trainer import Trainer
from onnx2torch import convert
from pathlib import Path
from typing import Any, Iterator
from torch import nn
from adversarial_training_box.database.experiment_tracker import ExperimentTracker


from adversarial_training_box.pipeline.training_module import TrainingModule

import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import torch.onnx
import shutil
import torch
import copy

from torchmetrics import Accuracy

class StandardTransferLearningModule():
    def __init__(self, base_path, experiment_tracker: ExperimentTracker = None):
        self.base_path = base_path
        self.experiment_tracker = experiment_tracker

    def retrain(self, model, dataloader, retrainings, base_path="./", max_epochs=1000, inp=True):
        if retrainings < 0:
            retrainings = len(list(model.children()))

        new_models = []
        # if we want to retrain 1 layer, we need to loop from 1 to 2 etc.
        for i in range(1, retrainings+1):
            trained_model = None
            while trained_model is None:
                new_model = copy.deepcopy(model)
                # reset last layers
                if new_model.reset_k_layers(i):
                    print(f"Starting retraining the last {i} layers")
                    new_model.only_train_k_layers(i)
                    trained_model = self.train(dataloader,new_model, f"{model.name}_retraining[{i}]", base_path, max_epochs, inp)
                else:
                    print(f"Skipping retraining the last {i} layers, as layer {type(list(model.children())[::-1][i-1])} does not have parameters")
                    break
            if trained_model is not None:
                new_models.append(trained_model)
        return new_models

    def transfer(self,model, dataloader, split_layers, out_features, dataset_name, base_path="./", max_epochs=1000, inp=True):
        new_models = []
        for split_layer in split_layers:
            new_model = copy.deepcopy(model)

            if new_model.reset_k_layers(split_layer):
                print(f"Starting retraining the last {split_layer} layers")
                new_model.only_train_k_layers(split_layer)
                trained_model = self.train(dataloader,new_model, f"{model.name}_transfer_{dataset_name}_[{split_layer}]", base_path, max_epochs, inp)
            else:
                print(
                    f"Skipping retraining the last {split_layer} layers, as layer {type(list(model.children())[::-1][split_layers - 1])} does not have parameters")
                break
            if trained_model is not None:
                new_models.append(trained_model)
        return new_models


    def train(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module, name, base_path="./", max_epochs=1000, inp=None):
        patience = 50

        Path(base_path).mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project="TransferLearning")

        checkpoint = Path(f"{base_path}/model_checkpoints/{name}")
        checkpoint.mkdir(parents=True, exist_ok=True)

        weight_averaging = StochasticWeightAveraging(swa_lrs=1e-2)
        # checkpointer = ModelCheckpoint(checkpoint, save_top_k=-1, monitor="val_acc")
        early_stopping = EarlyStopping("val_acc", mode="max", patience=patience)

        trainer = Trainer(logger=logger, gradient_clip_val=0.5,
                        callbacks=[weight_averaging, early_stopping], max_epochs=max_epochs)
        trainer.fit(network, data_loader)

        if trainer.current_epoch <= patience + 1:  # vanishing gradients detected
            print(f"Detected vanishing gradients for model {name}")
            return None

        trainer.test(network, data_loader)

        save_path = Path(f"{base_path}/trained_models")
        save_path.mkdir(parents=True, exist_ok=True)

        if inp is None:
            torch.save(network, save_path / name)
        else:
            if isinstance(inp, bool) and inp:
                inp = data_loader.dummy_input()
            torch.onnx.export(network, inp, str((save_path / name).with_suffix(".onnx")), export_params=True, do_constant_folding=True)

        return network


    def extract_trained_models(self, path, save_path):
        """Extract the trained model files of all models into one directory"""
        if not isinstance(path, Path):
            path = Path(path)
        if not isinstance(save_path, Path):
            save_path = Path(save_path)

        save_path.mkdir(exist_ok=True, parents=True)

        for idir in path.iterdir():
            trained_path = idir / "trained_models"
            if not trained_path.exists():
                print(f"Skipping {trained_path} as it does not exist")
                continue

            for model in trained_path.iterdir():
                if not model.is_file() or not model.suffix == ".onnx":
                    print(f"Skipping {model}")
                    continue
                shutil.copy2(model, save_path)

    def to_onnx(self, path, sample_input):
        if not isinstance(path, Path):
            path = Path(path)

        for model_path in path.iterdir():
            model = torch.load(model_path)
            torch.onnx.export(model,
                            sample_input,
                            str(model_path.with_suffix(".onnx")))


    def load_pretrained_model(self, model_name, dataset=False, base_path=Path("../nn-verification-assessment/networks"), convert_func=lambda x: x):
        """load models from a structure like https://github.com/marti-mcfly/nn-verification-assessment"""
        if not isinstance(base_path, Path):
            base_path = Path(base_path)

        if isinstance(dataset, bool) and not dataset:
            return convert_func(convert((base_path / model_name).with_suffix(".onnx")))
        elif isinstance(dataset, str):
            return convert_func(convert((base_path / dataset / model_name).with_suffix(".onnx")))


    def get_model_name(self, network_id, dataset, base_path=Path("../nn-verification-assessment/networks"), complete_path=False):
        """Look up the model path to the corresponding model Name"""
        if not isinstance(base_path, Path):
            base_path = Path(base_path)

        df = pd.read_csv(base_path / f"network_ids-{dataset}.csv", delimiter=";")
        if complete_path:
            return base_path / dataset / df.loc[df['network_id'] == network_id, 'network_filenames'].item()
        else:
            return df.loc[df['network_id'] == network_id, 'network_filenames'].item()


    def get_all_models(self, dataset, base_path=Path("../nn-verification-assessment/networks"), complete_path=False):
        """Generate a list of all models for the dataset"""
        if not isinstance(base_path, Path):
            base_path = Path(base_path)

        df = pd.read_csv(base_path / f"network_ids-{dataset}.csv", delimiter=";")

        if complete_path:
            return
        else:
            return [row[0] for row in df.groupby(["network_filenames"])["network_filenames"]]

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
        self.model.requires_grad_()
        if k < 0:  # train all layers
            return

        for layer in list(self.model.children())[:-k]:
            if hasattr(layer, 'reset_parameters'):
                layer.requires_grad_(False)

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
                # layer.reset_parameters()
        return could_reset

    def children(self) -> Iterator['Module']:
        return self.model.children()

    # def on_after_backward(self) -> None:
        # if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
        # if self.last_weights is not None:
        #     print(self.last_weights)
        #     print(self.last_weights - torch.Tensor(list(list(self.model.children())[-1].parameters())[0].clone()))
        # self.last_weights = torch.Tensor(list(list(self.model.children())[-1].parameters())[0].clone())
