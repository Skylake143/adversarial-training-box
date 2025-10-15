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

from torchvision import transforms
import os

from adversarial_training_box.datasets.cifar10 import CIFAR10
from adversarial_training_box.datasets.mnist import MNIST
from adversarial_training_box.datasets.emnist import EMNIST

from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from adversarial_training_box.models.mnist_net_256x2 import MNIST_NET_256x2
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
from adversarial_training_box.adversarial_attack.auto_attack_module import AutoAttackModule

from adversarial_training_box.pipeline.standard_transfer_learning_module import StandardTransferLearningModule, LightningWrapper

if __name__ == "__main__":
    training_parameters = AttributeDict(
        learning_rate = 0.002,
        weight_decay = 1e-5,
        scheduler_step_size=3,
        scheduler_gamma=0.96,
        attack_epsilon=0.3, 
        patience_epochs=5, 
        batch_size=2048,
        num_workers = 5)
    
    dataloader = EMNIST("adversarial_training_box/datasets", training_parameters.batch_size, download=True, transform=transforms.ToTensor(), num_workers=training_parameters.num_workers)

    # Transfer learning parameters
    source_model_name = "mnist_net_256x2.onnx"
    source_model_path =Path("generated/mnist_net_256x2-pgd-training/TransferTryout")
    split_layer = [1]

    # Transfer learning module 
    transfer_learning_module = StandardTransferLearningModule(source_model_path)

    # Network converter to adapt to target dataset
    def mnist256x2_converter(network: MNIST_NET_256x2):
        setattr(network, 'layer3/Gemm', torch.nn.Linear(256, 47))
        return network
    
    if not isinstance(source_model_name, Path):
        source_model_name = Path(source_model_name)

    try:
        model = LightningWrapper(transfer_learning_module.load_pretrained_model(source_model_name, False, base_path=source_model_path, convert_func=mnist256x2_converter), source_model_name.with_suffix(""))
        
        # Debug: Check which parameters require gradients
        print("Parameters requiring gradients:")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        
        # 47 is the number of classes of emnist
        models = transfer_learning_module.transfer(model, dataloader, split_layer, 47, "emnist", f"./transfer/emnist_{source_model_name}_({time.time()})",max_epochs=50)
    except Exception as e:
        print(f"Failed to train model {source_model_name} with exception {e}")

