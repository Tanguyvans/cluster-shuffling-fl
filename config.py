import os
from pathlib import Path

# Dataset root directory - uses local dataset/ directory in project
DATASET_ROOT = os.path.join(os.path.dirname(__file__), 'dataset')

# Specific dataset paths
DATASET_PATHS = {
    'cifar10': os.path.join(DATASET_ROOT, 'cifar10'),
    'cifar100': os.path.join(DATASET_ROOT, 'cifar-100-python'),
    'ffhq128': os.path.join(DATASET_ROOT, 'ffhq_dataset'),
    'caltech256': os.path.join(DATASET_ROOT, 'caltech256'),
}

settings = {
    "name_dataset": "cifar10",  # "cifar10", "cifar100", "ffhq128", "caltech256"
    "data_root": DATASET_ROOT,  # Root directory for all datasets
    "arch": "simplenet",        # simplenet, mobilenet, resnet18, shufflenet, squeezenet, efficientnet
    "pretrained": False,        # Use pretrained weights
    "input_size": 32,           # Input size (32 for CIFAR, 128 for FFHQ)
    "patience": 3,
    "batch_size": 32,           # Normal batch size for training
    "n_epochs": 5,              # Multiple epochs for proper training
    "num_classes": 10,          # 10 classes for CIFAR-10

    # Normal training settings
    "single_batch_training": False,   # Use full dataset
    "balanced_class_training": False, # Normal data distribution
    "max_samples_per_client": None,   # No limit on samples
    "number_of_nodes": 1,
    "number_of_clients_per_node": 6,
    "min_number_of_clients_in_cluster": 3,

    "check_usefulness": False,
    "coef_useful": 1.05,
    "tolerance_ceil": 0.08,

    "n_rounds": 10,             # Normal number of rounds
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,
    "weight_decay": 1e-4,
    "choice_scheduler": "StepLR",
    "step_size": 5,
    "gamma": 0.1,

    "diff_privacy": False,      # Disable DP for baseline
    "noise_multiplier": 0.1,
    "max_grad_norm": 0.5,
    "delta": 1e-5,
    "epsilon": 5.0,

    "clustering": False,        # Disable clustering for baseline
    "type_ss": "additif",
    "threshold": 3,
    "m": 3,
    "ts": 5,

    # Gradient saving disabled for normal training
    "save_gradients": False,                  # Disable gradient saving
    "save_gradients_rounds": [],              # No rounds to save
    "use_pytorch_smpc": True,                  # Use pure PyTorch SMPC (no NumPy)
    "aggregation_method": "weights",         # "weights" or "gradients" - what to use for SMPC/aggregation
                                               # "gradients": More private, smaller data, better for attacks
                                               # "weights": Traditional FL, larger data, current implementation

    # Gradient pruning/compression (Deep Gradient Compression)
    "gradient_pruning": {
        "enabled": True,                       # Enable gradient pruning for communication efficiency
        "keep_ratio": 0.1,                     # Fraction of gradients to keep (0.1 = 10%, 90% compression)
        "momentum_factor": 0.9,                # Momentum for velocity buffer (standard: 0.9)
        "use_momentum_correction": True,       # Enable DGC momentum correction (recommended)
        "sample_ratio": 0.01,                  # Sampling ratio for threshold estimation (1%)
    },
    
    "save_figure": True,
    "matrix_path": "results/CFL/matrix_path",
    "roc_path": "results/CFL/roc_path",
    
    # Aggregation method configuration
    "aggregation": {
        "method": "fedavg",  # Options: fedavg, krum, multi_krum, trimmed_mean, median, fltrust
        "krum_malicious": 1,  # Number of malicious clients to tolerate (for krum)
        "multi_krum_keep": 3,  # Number of clients to keep (for multi_krum)
        "trim_ratio": 0.2,  # Fraction to trim (for trimmed_mean)
        "fltrust_root_size": 5000,  # Size of server's root dataset for FLTrust
        "fltrust_learning_rate": 0.01,  # Learning rate for FLTrust server model
        "fltrust_server_epochs": 5,  # Number of epochs for server training
    },

    # Poisoning attack configuration
    "poisoning_attacks": {
        "enabled": False,                        # Disable attacks for normal training
        "malicious_clients": [],                 # No malicious clients
        "attack_type": "noise",                  # Attack type: labelflip, noise, signflip, alie, ipm, backdoor
        "attack_intensity": 0.0,                 # No attack intensity
        "attack_rounds": None,                   # Specific rounds to attack (None = all rounds)
        "attack_frequency": 0.0,                 # No attacks
        
        # Attack-specific configurations
        "labelflip_config": {
            "flip_type": "targeted",               # targeted, random, all_to_one
            "source_class": None,                  # Class to flip from (None = all classes)
            "target_class": 0,                     # Target class for flipping
            "num_classes": 10
        },
        
        "noise_config": {
            "noise_type": "gaussian",              # gaussian, uniform, laplacian
            "noise_std": 0.1,                      # Standard deviation of noise
            "target_layers": None,                 # Layers to target (None = all)
            "adaptive_noise": False                # Scale noise based on parameter magnitude
        },
        
        "signflip_config": {
            "flip_strategy": "random",             # random, all, selective
            "target_layers": None,                 # Layers to target (None = all)
            "flip_probability": 0.3,               # Probability of flipping each gradient
            "magnitude_scaling": 1.0               # Scale factor for flipped gradients
        },
        
        "alie_config": {
            "deviation_type": "sign",              # sign, std, mean
            "aggregation_type": "mean",            # Expected aggregation method
            "epsilon": 0.1,                        # Small deviation parameter
            "num_malicious": 1                     # Number of malicious clients
        },
        
        "ipm_config": {
            "target_level": "client",              # "client" or "cross_cluster"
            "lambda_param": 0.2,                   # Manipulation strength (λ in formula: -λ * mean(benign))
        },
        
        "backdoor_config": {
            "trigger_type": "pixel_pattern",       # pixel_pattern, square, cross, random_noise
            "trigger_size": 3,                     # Size of trigger pattern
            "trigger_position": "bottom_right",    # bottom_right, top_left, center, random
            "trigger_value": 1.0,                  # Trigger pixel intensity
            "backdoor_label": 0,                   # Target label for backdoor
            "poison_all_classes": True             # Poison samples from all classes
        }
    }
}
