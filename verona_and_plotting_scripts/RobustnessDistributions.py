# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib.util
import logging
from pathlib import Path
import random
import argparse
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import ada_verona.util.logger as logger
from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from autoverify.verifier import AbCrown, Nnenum
from ada_verona.verification_module.auto_verify_module import AutoVerifyModule
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from ada_verona.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.verification_module.attacks.pgd_attack import PGDAttack
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator

logger.setup_logging(level=logging.INFO)

torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser(description='Robustness distribution script')
    parser.add_argument('--directory_path', type=str, default='"examples/CNNYangBig/emnist_cnn_yang_big-pgd-training_21-10-2025+13_06"', help='Source model path')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'EMNIST', 'CIFAR10', 'CIFAR100'], default='EMNIST', help='Dataset to use (MNIST, EMNIST, CIFAR10, CIFAR100)')
    args = parser.parse_args()

    # Load dataset based on argument
    if args.dataset == 'MNIST':
        #dataset_size = 20000
        torch_dataset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=torchvision.transforms.ToTensor())
        epsilon_list = list(np.arange(0.00, 0.800, 0.005))
        # total_dataset_size = len(torch_dataset)
        # subset_indices = random.sample(range(total_dataset_size), dataset_size)
        # subset_torch_dataset = torch.utils.data.Subset(torch_dataset, subset_indices)
        subset_torch_dataset = torch_dataset
        # Data bounds for unnormalized MNIST
        data_lb = 0.0
        data_ub = 1.0
        num_classes = 10
        epsilon_scale_factor = 1.0
    elif args.dataset == 'EMNIST':
        #dataset_size = 20000
        torch_dataset = torchvision.datasets.EMNIST('../data', split="balanced", train=False, download=True, transform=torchvision.transforms.ToTensor())
        epsilon_list = list(np.arange(0.00, 0.800, 0.005))
        # Create random indices for sampling
        # total_dataset_size = len(torch_dataset)
        # subset_indices = random.sample(range(total_dataset_size), dataset_size)
        # subset_torch_dataset = torch.utils.data.Subset(torch_dataset, subset_indices)
        subset_torch_dataset = torch_dataset
        # Data bounds for unnormalized EMNIST
        data_lb = 0.0
        data_ub = 1.0
        num_classes = 47
        epsilon_scale_factor = 1.0
    elif args.dataset == 'CIFAR10':
        #dataset_size = 500
        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std = [0.2470, 0.2435, 0.2616]
        mean_std = sum(cifar_std) / len(cifar_std)
        
        cifar_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
        torch_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=True, transform=cifar_transform)
        # Scale epsilon to normalized space (divide by mean_std)
        epsilon_list = [eps / mean_std for eps in np.arange(0.00, 28/256, 0.001)]
        #Create random indices for sampling
        # total_dataset_size = len(torch_dataset)
        # subset_indices = random.sample(range(total_dataset_size), dataset_size)
        # subset_torch_dataset = torch.utils.data.Subset(torch_dataset, subset_indices)
        subset_torch_dataset = torch_dataset
        # Data bounds for normalized CIFAR-10
        data_lb = -2.5
        data_ub = 2.5
        num_classes = 10
        epsilon_scale_factor = mean_std
    elif args.dataset == 'CIFAR100':
        #dataset_size = 500
        cifar_mean = [0.5071, 0.4865, 0.4409]
        cifar_std = [0.2673, 0.2564, 0.2762]
        mean_std = sum(cifar_std) / len(cifar_std)
        
        cifar_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
        torch_dataset = torchvision.datasets.CIFAR100('../data', train=False, download=True, transform=cifar_transform)
        # Scale epsilon to normalized space (divide by mean_std)
        epsilon_list = [eps / mean_std for eps in np.arange(0.00, 28/256, 0.001)]
        # Create random indices for sampling
        # total_dataset_size = len(torch_dataset)
        # subset_indices = random.sample(range(total_dataset_size), dataset_size)
        # subset_torch_dataset = torch.utils.data.Subset(torch_dataset, subset_indices)
        subset_torch_dataset = torch_dataset
        # Data bounds for normalized CIFAR-100
        data_lb = -2.5
        data_ub = 2.5
        num_classes = 100
        epsilon_scale_factor = mean_std
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    experiment_name = "pgd"
    timeout = 1000
    experiment_repository_path = Path(args.directory_path) / "results" # Path("../CNNYangBig/results")
    network_folder = Path(args.directory_path)

    dataset = PytorchExperimentDataset(dataset=subset_torch_dataset)

    file_database = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    file_database.initialize_new_experiment(experiment_name)

    file_database.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            network_folder=str(network_folder),
            dataset=args.dataset,
            dataset_details=str(dataset),
            timeout=timeout,
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )

    property_generator = One2AnyPropertyGenerator(number_classes=num_classes, data_lb=data_lb, data_ub=data_ub)
    verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=40, data_lb=data_lb, data_ub=data_ub))

    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=epsilon_list.copy(), verifier=verifier)

    network_list = sorted(file_database.get_network_list(), key=lambda network: network.name)

    for network in network_list:
        for data_point in dataset:
            verification_context = file_database.create_verification_context(network, data_point, property_generator)

            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

            if epsilon_scale_factor != 1.0:
                epsilon_value_result.epsilon *= epsilon_scale_factor
                epsilon_value_result.smallest_sat_value *= epsilon_scale_factor

            file_database.save_result(epsilon_value_result)

    file_database.save_plots()

    # Copy saved results to experiment_repository_path
    src_results_path = file_database.get_results_path()
    dst_results_path = experiment_repository_path
    
    # Create destination directory if it doesn't exist
    dst_results_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from source results to destination
    for item in src_results_path.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_results_path / item.name)
        elif item.is_dir():
            shutil.copytree(item, dst_results_path / item.name, dirs_exist_ok=True)

if __name__ == "__main__":
    main()