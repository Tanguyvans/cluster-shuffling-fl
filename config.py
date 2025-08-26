import os
from pathlib import Path

# Dataset root directory - uses ~/data as the centralized location
DATASET_ROOT = os.path.expanduser('~/data')

# Specific dataset paths
DATASET_PATHS = {
    'cifar10': os.path.join(DATASET_ROOT, 'cifar10'),
    'cifar100': os.path.join(DATASET_ROOT, 'cifar-100-python'),
    'ffhq': os.path.join(DATASET_ROOT, 'ffhq_dataset'),
    'ffhq128': os.path.join(DATASET_ROOT, 'ffhq_dataset'),
    'caltech256': os.path.join(DATASET_ROOT, 'caltech256'),
}

settings = {
    "name_dataset": "cifar10",  # "cifar10" or "cifar100" or "ffhq128"
    "data_root": DATASET_ROOT,  # Root directory for all datasets
    "arch": "convnet",  # "mobilenet" or "resnet18" or "shufflenet"
    "pretrained": False,
    "patience": 3,
    "batch_size": 10,
    "n_epochs": 3,
    
    # Single batch training for inference attacks
    "single_batch_training": True,  # Set to True to train on only one batch per epoch
    "balanced_class_training": True,  # Ensure each client gets one sample per class (like template)
    "number_of_nodes": 1,
    "number_of_clients_per_node": 3,
    "min_number_of_clients_in_cluster": 3,

    "check_usefulness": False,
    "coef_useful": 1.05,   # 1.05
    "tolerance_ceil": 0.08,

    "poisoned_number": 0,
    "n_rounds": 3,
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,              # Back to original LR
    "choice_scheduler": "StepLR",  # Re-enable StepLR to test the fix
    "step_size": 3,
    "gamma": 0.5,

    "diff_privacy": False,
    "noise_multiplier": 0.1,  
    "max_grad_norm": 0.5,     
    "delta": 1e-5,
    "epsilon": 5.0,           

    "clustering": False,       # RE-ENABLE MPC - Testing the fix!
    "type_ss": "additif",
    "threshold": 3,
    "m": 3,
    "ts": 5,                  

    # New settings for PyTorch-based SMPC and gradient saving
    "save_gradients": True,                    # Enable gradient saving for attacks
    "save_gradients_rounds": [1, 2, 3],       # Which rounds to save gradients
    "use_pytorch_smpc": True,                  # Use pure PyTorch SMPC (no NumPy)
    "aggregation_method": "gradients",         # "weights" or "gradients" - what to use for SMPC/aggregation
                                               # "gradients": More private, smaller data, better for attacks
                                               # "weights": Traditional FL, larger data, current implementation
    
    "save_figure": True,
    "matrix_path": "results/CFL/matrix_path",
    "roc_path": "results/CFL/roc_path",
}
