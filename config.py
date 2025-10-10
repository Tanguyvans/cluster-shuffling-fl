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
    "name_dataset": "ffhq128",  # "cifar10", "cifar100", "ffhq128" (use ffhq128 for GIFD GAN attacks)
    "data_root": DATASET_ROOT,  # Root directory for all datasets
    "arch": "resnet18",  # ResNet18 matching train_ffhq_resnet.py
    "pretrained": False,  # Don't use ImageNet pretrained weights (train from scratch like train_ffhq_resnet.py)
    "input_size": 32,  # 32x32 input size to match train_ffhq_resnet.py and GIAS core
    "patience": 3,
    "batch_size": 1,  # Batch size of 1 to match train_ffhq_resnet.py
    "n_epochs": 1,     # Single epoch like train_ffhq_resnet.py
    "num_classes": 6,  # 6 age groups matching train_ffhq_resnet.py (0-10, 10-20, 20-30, 30-40, 40-50, 50+)

    # Single batch training for inference attacks - MATCHING train_ffhq_resnet.py
    "single_batch_training": True,   # Train on only one batch per epoch
    "balanced_class_training": True, # Ensure each client gets balanced samples
    "max_samples_per_client": 1,     # Exactly 1 sample per client (matching train_ffhq_resnet.py)
    "number_of_nodes": 1,
    "number_of_clients_per_node": 6,
    "min_number_of_clients_in_cluster": 3,

    "check_usefulness": False,
    "coef_useful": 1.05,   # 1.05
    "tolerance_ceil": 0.08,

    "n_rounds": 3,  # Fewer rounds for focused gradient attack evaluation
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,               # Learning rate matching train_ffhq_resnet.py
    "weight_decay": 1e-4,      # Weight decay matching train_ffhq_resnet.py
    "choice_scheduler": "StepLR",
    "step_size": 2,            # Step size matching train_ffhq_resnet.py
    "gamma": 0.1,              # Gamma matching train_ffhq_resnet.py

    "diff_privacy": False,
    "noise_multiplier": 0.1,  
    "max_grad_norm": 0.5,     
    "delta": 1e-5,
    "epsilon": 5.0,           

    "clustering": False,       # Testing Krum without SMPC first
    "type_ss": "additif",
    "threshold": 3,
    "m": 3,
    "ts": 5,                  

    # New settings for PyTorch-based SMPC and gradient saving
    "save_gradients": True,                    # Enable gradient saving for attacks
    "save_gradients_rounds": [1, 2, 3],       # Which rounds to save gradients (all rounds for small dataset)
    "use_pytorch_smpc": True,                  # Use pure PyTorch SMPC (no NumPy)
    "aggregation_method": "weights",         # "weights" or "gradients" - what to use for SMPC/aggregation
                                               # "gradients": More private, smaller data, better for attacks
                                               # "weights": Traditional FL, larger data, current implementation
    
    "save_figure": True,
    "matrix_path": "results/CFL/matrix_path",
    "roc_path": "results/CFL/roc_path",
    
    # Aggregation method configuration
    "aggregation": {
        "method": "krum",  # Options: fedavg, krum, multi_krum, trimmed_mean, median, fltrust
        "krum_malicious": 1,  # Number of malicious clients to tolerate (for krum)
        "multi_krum_keep": 3,  # Number of clients to keep (for multi_krum)
        "trim_ratio": 0.2,  # Fraction to trim (for trimmed_mean)
        "fltrust_root_size": 5000,  # Size of server's root dataset for FLTrust (increased)
        "fltrust_learning_rate": 0.01,  # Learning rate for FLTrust server model
        "fltrust_server_epochs": 5,  # Number of epochs for server training
    },
    
    # Poisoning attack configuration
    "poisoning_attacks": {
        "enabled": False,                            # Enable poisoning attacks
        "malicious_clients": ["c0_1"],              # List of malicious client IDs (e.g., ["c0_1", "c0_2"])
        "attack_type": "noise",                # Attack type: labelflip, noise, signflip, alie, ipm, backdoor
        "attack_intensity": 1.0,                   # Attack strength (0.0 to 1.0)
        "attack_rounds": None,                     # Specific rounds to attack (None = all rounds)
        "attack_frequency": 1.0,                   # Probability of attacking each round
        
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
