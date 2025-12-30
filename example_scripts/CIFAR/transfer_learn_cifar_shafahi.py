from random import shuffle
import copy
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.nn as nn
from pathlib import Path
import optuna
import argparse
from optuna.trial import TrialState
from adversarial_training_box.pipeline.early_stopper import EarlyStopper

from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from adversarial_training_box.models.emnist_net_256x2 import EMNIST_NET_256x2
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
from adversarial_training_box.adversarial_attack.auto_attack_module import AutoAttackModule
from torchvision.models.resnet import BasicBlock, Bottleneck

import torch
import torch.nn.functional as F

class LwFLoss(nn.Module):
    """Learning without Forgetting Loss Implementation"""
    def __init__(self, lambda_lwf=1.0, temperature=4.0):
        super().__init__()
        self.lambda_lwf = lambda_lwf
        self.temperature = temperature
        
    def forward(self, new_outputs, old_outputs, targets, old_features=None, new_features=None):
        # Standard cross-entropy loss for new task
        ce_loss = F.cross_entropy(new_outputs, targets)
        
        # Knowledge distillation loss on outputs - only if dimensions match
        kd_loss = 0
        if old_outputs.size(-1) == new_outputs.size(-1):
            old_outputs_soft = F.softmax(old_outputs / self.temperature, dim=1)
            new_outputs_soft = F.log_softmax(new_outputs / self.temperature, dim=1)
            kd_loss = F.kl_div(new_outputs_soft, old_outputs_soft, reduction='batchmean')
            kd_loss *= (self.temperature ** 2)
        
        # Feature representation preservation (similar to the TensorFlow version)
        feat_loss = 0
        if old_features is not None and new_features is not None:
            # L1 norm difference between old and new features (matching TF version)
            feat_loss = torch.mean(torch.norm(new_features - old_features, p=1, dim=1))
        
        total_loss = ce_loss + self.lambda_lwf * (kd_loss + feat_loss)
        return total_loss, ce_loss, kd_loss, feat_loss

def extract_features(model, x, feature_layer_name='avg_pool'):
    """Extract intermediate features from model"""
    features = None
    def hook_fn(module, input, output):
        nonlocal features
        features = output.flatten(1)  # Flatten to (batch_size, -1)
    
    # Register hook on the specified layer
    for name, module in model.named_modules():
        if feature_layer_name in name:
            handle = module.register_forward_hook(hook_fn)
            break
    
    with torch.no_grad():
        _ = model(x)
    
    if 'handle' in locals():
        handle.remove()
    
    return features.detach() if features is not None else None

# Modified training module for LwF
class LwFTrainingModule:
    def __init__(self, criterion, lwf_criterion, source_model, lambda_lwf=1.0):
        self.criterion = criterion
        self.lwf_criterion = lwf_criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source_model = source_model.eval().to(self.device)  # Keep source model frozen and on correct device
        self.lambda_lwf = lambda_lwf
        
    def train(self, data_loader, network, optimizer, experiment_tracker=None):
        network.train()
        network = network.to(self.device)
        self.source_model.eval()
        
        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        kd_loss_sum = 0.0
        feat_loss_sum = 0.0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass through both models
            new_outputs = network(data)
            
            with torch.no_grad():
                old_outputs = self.source_model(data)
                old_features = extract_features(self.source_model, data)
            
            new_features = extract_features(network, data)
            
            # Compute LwF loss
            total_loss, ce_loss, kd_loss, feat_loss = self.lwf_criterion(
                new_outputs, old_outputs, target, old_features, new_features
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            batch_size = target.size(0)
            total_loss_sum += total_loss.item() * batch_size
            ce_loss_sum += ce_loss.item() * batch_size
            kd_loss_sum += (kd_loss.item() if torch.is_tensor(kd_loss) else kd_loss) * batch_size
            feat_loss_sum += (feat_loss.item() if torch.is_tensor(feat_loss) else feat_loss) * batch_size
            total_samples += batch_size
            
            if experiment_tracker:
                experiment_tracker.log({
                    "train_loss": total_loss.item(),
                    "ce_loss": ce_loss.item(),
                    "kd_loss": kd_loss.item() if torch.is_tensor(kd_loss) else kd_loss,
                    "feat_loss": feat_loss.item() if torch.is_tensor(feat_loss) else feat_loss
                })
        
        # Calculate accuracy for compatibility with pipeline
        correct_predictions = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                outputs = network(data)
                predictions = outputs.max(1)[1]
                correct_predictions += (predictions == target).sum().item()
        
        train_accuracy = correct_predictions / total_samples
        
        # Return in the format expected by pipeline (train_accuracy, robust_accuracy)
        return train_accuracy, None

def get_resnet_blocks(model):
    """Extract all ResNet blocks"""

    blocks = []
    # For the ResNet architecture, that we use
    if hasattr(model, 'conv2_x'):
        blocks.extend([
            ('conv1', model.conv1),
            ('conv2_x', model.conv2_x),
            ('conv3_x', model.conv3_x),
            ('conv4_x', model.conv4_x),
            ('conv5_x', model.conv5_x),
            ('avg_pool', model.avg_pool),
            ('fc', model.fc)
        ])
    else:
        # Fallback for other ResNet structures
        for name, module in model.named_children():
            blocks.append((name, module))
    return blocks

def reset_last_k_layers(model, k):
    """Reset the parameters of the last k layers of a model"""
    blocks = get_resnet_blocks(model)

    if k > len(blocks):
        raise ValueError(f"k ({k}) cannot be larger than the number of layers ({len(blocks)})")
    
    if k <= 0:
        raise ValueError(f"k must be positive (got {k}). For transfer learning, you must retrain at least 1 block.")
    
    # Reset the last k blocks
    for name, block in blocks[-k:]:
        print(f"  Resetting: {name}")
        for module in block.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

def freeze_except_last_k_layers(model, k):
    """Freeze all parameters except the last k layers"""
    blocks = get_resnet_blocks(model)

    if k > len(blocks):
        raise ValueError(f"k ({k}) cannot be larger than the number of layers ({len(blocks)})")
    
    if k <= 0:
        raise ValueError(f"k must be positive (got {k}). For transfer learning, you must retrain at least 1 block.")

    for param in model.parameters():
        param.requires_grad= False

    # Unfreeze only the last k blocks
    for name, block in blocks[-k:]:
        print(f"  Unfreezing: {name}")
        for param in block.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Transfer Learning script with configurable network and experiment name')
    parser.add_argument('--source_model_path', type=str, default='"generated/BachelorThesisRuns/cnn_yang_big-pgd-training_21-10-2025+12_40/cnn_yang_big.pth"',
                       help='Source model path')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name (default: {source_model_path}-transfer-learning)')
    parser.add_argument('--retraining_layers', type=int, default=1,
                       help='Indicate number of layers to retrain')
    args = parser.parse_args()  # Add this line to actually parse the arguments

    training_parameters = AttributeDict(
        learning_rate = 0.1,
        weight_decay = 5e-4,
        momentum = 0.9,
        scheduler_milestones=[60, 120, 160],
        scheduler_gamma=0.2,
        patience_epochs=6,
        overhead_delta=0.0,
        batch_size=256)
    
    source_model_path = Path(args.source_model_path)
    experiment_name = args.experiment_name
    retraining_layers = args.retraining_layers

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Source model
    source_model = torch.load(source_model_path, map_location='cpu')
    source_model_copy = copy.deepcopy(source_model)
    source_model_copy.eval().to(device)

    # Create separate LwF source model that keeps original architecture
    source_model_for_lwf = copy.deepcopy(source_model)
    source_model_for_lwf.eval().to(device)


    # Network converter to adapt to target domain
    def convert_last_layer(network, num_classes=100):
        layers = list(network.named_modules())
        last_layer_name = layers[-1][0]
        last_layer_module = layers[-1][1]

        in_features = last_layer_module.in_features
        has_bias = last_layer_module.bias is not None
        new_layer = torch.nn.Linear(in_features, num_classes, bias=has_bias)

        # Replace the last layer in the network
        setattr(network, last_layer_name, new_layer)
        
        return network
    
    # Convert source model for target task
    converted_model = convert_last_layer(source_model_copy)

    # Reset/freeze layers based on retraining strategy
    if retraining_layers > 0:
        print(f"Resetting last {retraining_layers} layers")
        reset_last_k_layers(converted_model, retraining_layers)
        print(f"Freezing all except last {retraining_layers} layers") 
        freeze_except_last_k_layers(converted_model, retraining_layers)

    cifar_mean = [0.5071, 0.4865, 0.4409]
    cifar_std = [0.2673, 0.2564, 0.2762]
    
    normalize = transforms.Normalize(mean=cifar_mean,std=cifar_std)

    mean_std = sum(cifar_std) / len(cifar_std)

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    num_classes = 100
    full_train_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)
    full_validation_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=test_transform)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(43)

    # Split with same indices for both
    train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, validation_dataset = torch.utils.data.random_split(full_validation_dataset, [train_size, val_size], generator=generator)

    test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=test_transform)
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)    

    # Setup optimizer and scheduler
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, converted_model.parameters()),
                         lr=training_parameters.learning_rate,
                         momentum=training_parameters.momentum,
                         weight_decay=training_parameters.weight_decay)
    
    scheduler = MultiStepLR(optimizer, 
                           milestones=training_parameters.scheduler_milestones,
                           gamma=training_parameters.scheduler_gamma)
    
    # Setup early stopper
    early_stopper = EarlyStopper(patience=training_parameters.patience_epochs,
                                delta=training_parameters.overhead_delta)

    # Training configuration
    criterion = nn.CrossEntropyLoss()

    # Validation module
    validation_module = StandardTestModule(criterion=criterion)

    # Training modules stack
    # Setup LwF components
    lwf_criterion = LwFLoss(lambda_lwf=1.0, temperature=4.0)
    lwf_training_module = LwFTrainingModule(
        criterion=criterion,
        lwf_criterion=lwf_criterion,
        source_model=source_model_for_lwf,
        lambda_lwf=1.0
    )
    
    # Modified training stack with LwF
    training_stack = []
    training_stack.append((200, lwf_training_module))

    # Testing modules stack
    testing_stack = [StandardTestModule(),
        StandardTestModule(attack=FGSMAttack(), epsilon=2/255/mean_std),
        StandardTestModule(attack=FGSMAttack(), epsilon=4/255/mean_std),
        StandardTestModule(attack=FGSMAttack(), epsilon=8/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=2/255/mean_std/4, number_iterations=20, random_init=True), epsilon=2/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=4/255/mean_std/4, number_iterations=20, random_init=True), epsilon=4/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=8/255/mean_std/4, number_iterations=20, random_init=True), epsilon=8/255/mean_std),
    ]
    
    # Convert complex objects to JSON-serializable format
    def serialize_training_stack(stack):
        return [{"epochs": epochs, "module_type": type(module).__name__, 
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for epochs, module in stack]
    
    def serialize_testing_stack(stack):
        return [{"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for module in stack]
    
    def serialize_validation_module(module):
        return {"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)}

    # Handle experiment name
    if experiment_name is None:
        experiment_name = f"{source_model_path.stem}-transfer-learning"

    training_objects = AttributeDict(criterion=str(criterion), 
                                     optimizer=str(optimizer), 
                                     network=str(converted_model), 
                                     scheduler=str(scheduler), 
                                     training_stack=serialize_training_stack(training_stack),
                                     testing_stack=serialize_testing_stack(testing_stack),
                                     validation_module=serialize_validation_module(validation_module))

    # Setup experiment
    experiment_tracker = ExperimentTracker(experiment_name, Path("./generated"), login=True)
    experiment_tracker.initialize_new_experiment(f"TL{retraining_layers}_shafahi", training_parameters=training_parameters | training_objects)
    pipeline = Pipeline(experiment_tracker, training_parameters, criterion, optimizer, scheduler)
    
    # Train
    pipeline.train(train_loader, converted_model, training_stack, early_stopper=early_stopper, 
                   validation_loader=validation_loader,
                   validation_module=validation_module
                   )
    
    # Test
    network = experiment_tracker.load_trained_model(converted_model)
    pipeline.test(network, test_loader, testing_stack=testing_stack)
    experiment_tracker.export_to_onnx(network, test_loader)

    # Finish logging
    experiment_tracker.finish()