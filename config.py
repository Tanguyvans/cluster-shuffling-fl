import os
from pathlib import Path

# Dataset root directory - uses ~/data as the centralized location
DATASET_ROOT = os.path.expanduser('~/data')

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
    "arch": "resnet18",  # ResNet18 is optimal for FFHQ128 (128x128 images)
    "pretrained": False,  # Don't use ImageNet pretrained weights for age classification
    "patience": 3,
    "batch_size": 4,  # Smaller batch size for 128x128 images (memory constraints)
    "n_epochs": 3,
    
    # Single batch training for inference attacks
    "single_batch_training": False,  # Set to True to train on only one batch per epoch
    "balanced_class_training": True,  # Ensure each client gets one sample per class (like template)
    "number_of_nodes": 1,
    "number_of_clients_per_node": 3,
    "min_number_of_clients_in_cluster": 3,

    "check_usefulness": False,
    "coef_useful": 1.05,   # 1.05
    "tolerance_ceil": 0.08,

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

    "clustering": True,       # RE-ENABLE MPC - Testing the fix!
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
    
    # Poisoning attack configuration
    "poisoning_attacks": {
        "enabled": False,                            # Enable poisoning attacks
        "malicious_clients": ["c0_1", "c0_2"],              # List of malicious client IDs (e.g., ["c0_1", "c0_2"])
        "attack_type": "labelflip",                # Attack type: labelflip, noise, signflip, alie, ipm, backdoor
        "attack_intensity": 0.2,                   # Attack strength (0.0 to 1.0)
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
