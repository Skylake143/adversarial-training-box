from random import shuffle
import time
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from pathlib import Path
import optuna
from optuna.trial import TrialState
from early_stopping_pytorch import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from onnx2torch import convert

from torchvision import transforms
import os
import copy

from adversarial_training_box.datasets.cifar10 import CIFAR10
from adversarial_training_box.datasets.mnist import MNIST
from adversarial_training_box.datasets.emnist import EMNIST

from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from adversarial_training_box.models.MNIST.mnist_relu_2_256 import MNIST_RELU_2_256
from adversarial_training_box.models.MNIST.cnn_yang_big import CNN_YANG_BIG
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
from adversarial_training_box.adversarial_attack.auto_attack_module import AutoAttackModule

from adversarial_training_box.pipeline.transfer_learning_pytorch_wrapper import LightningWrapper

def load_pretrained_model(model_name, dataset=False, base_path=Path("../nn-verification-assessment/networks"), convert_func=lambda x: x):
        """load models from a structure like https://github.com/marti-mcfly/nn-verification-assessment"""
        if not isinstance(base_path, Path):
            base_path = Path(base_path)

        if isinstance(dataset, bool) and not dataset:
            return convert_func(convert((base_path / model_name).with_suffix(".onnx")))
        elif isinstance(dataset, str):
            return convert_func(convert((base_path / dataset / model_name).with_suffix(".onnx")))

if __name__ == "__main__":
    training_parameters = AttributeDict(
        learning_rate = 0.002,
        weight_decay = 1e-5,
        scheduler_step_size=3,
        scheduler_gamma=0.96,
        attack_epsilon=0.3, 
        patience_epochs=5, 
        batch_size=2048,
        num_workers = 1,
        split_layer=2)
    
    # Datasets
    test_dataset = torchvision.datasets.EMNIST('../data', split="balanced",train=False, download=True, transform=torchvision.transforms.ToTensor())

    # Dataloaders
    dataloader = EMNIST("adversarial_training_box/datasets", training_parameters.batch_size, download=True, transform=transforms.ToTensor(), num_workers=training_parameters.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # Transfer learning parameters
    source_model_name = "cnn_yang_big.onnx"
    source_model_path =Path("generated/BachelorThesisRuns/cnn_yang_big-pgd-training_21-10-2025+12_40")
    final_model_name = "cnn_yang_big_transferred"

    # Setup experiment
    project = "cnn_yang_big"
    experiment_tracker = ExperimentTracker("cnn_yang_big-transfer-learning", Path("./generated"), login=True)
    logger = WandbLogger(project=project)
    experiment_tracker.initialize_new_experiment("", training_parameters=training_parameters, logger=logger)
    pipeline = Pipeline(experiment_tracker, training_parameters, criterion=None, optimizer=None, scheduler=None)

    # Testing modules stack
    testing_stack = [
        StandardTestModule(),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.1),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.2),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.3),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.1),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.2),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.3),
    ]

    # Network converter to adapt to target dataset
    def mnist_relu_2_256_converter(network: MNIST_RELU_2_256):
        setattr(network, 'layer3/Gemm', torch.nn.Linear(256, 47))
        return network
    
    def mnist_cnn_yang_converter(network: CNN_YANG_BIG):
        setattr(network, 'fc3/Gemm', torch.nn.Linear(200, 47))
        return network
    
    if not isinstance(source_model_name, Path):
        source_model_name = Path(source_model_name)

    # Source model
    source_model = LightningWrapper(load_pretrained_model(source_model_name, False, base_path=source_model_path, convert_func=mnist_cnn_yang_converter), final_model_name)
    source_model_copy = copy.deepcopy(source_model)

    
    # Transfer Learning
    trained_model = pipeline.transfer_learn(source_model=source_model_copy, dataloader=dataloader, split_layer=training_parameters.split_layer, max_epochs=50, logger=logger)
    
    # Test
    loaded_network = experiment_tracker.load_trained_model("cnn_yang_big_transferred")
    # TODO: fix that
    # experiment_tracker.export_to_onnx(network, test_loader)
    pipeline.test(loaded_network, test_loader, testing_stack=testing_stack)