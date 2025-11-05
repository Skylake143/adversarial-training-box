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