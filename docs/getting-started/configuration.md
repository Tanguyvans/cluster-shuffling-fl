# Configuration Guide

Complete reference for `config.py` settings.

## Overview

All system settings are centralized in `config.py`. Modify this file to configure:
- Dataset and model selection
- Training hyperparameters
- Privacy mechanisms (SMPC, DP, Clustering)
- Gradient pruning
- Poisoning attacks
- Aggregation methods

## Basic Settings

### Dataset Configuration

```python
"name_dataset": "cifar10",  # Dataset to use
"data_root": DATASET_ROOT,  # Root directory for datasets
"num_classes": 10,          # Number of classes (auto-set for standard datasets)
```

**Available Datasets:**
- `cifar10`: CIFAR-10 (32×32, 10 classes)
- `cifar100`: CIFAR-100 (32×32, 100 classes)
- `ffhq128`: FFHQ faces (128×128, 6 age groups)
- `caltech256`: Caltech256 objects (256 classes)

### Model Architecture

```python
"arch": "simplenet",        # Model architecture
"pretrained": False,        # Use pretrained weights
"input_size": 32,           # Input image size (32 for CIFAR, 128 for FFHQ)
```

**Available Architectures:**
- `simplenet`: Lightweight CNN (fast, good for testing)
- `mobilenet`: MobileNetV2 (efficient)
- `resnet18`: ResNet18 (accurate)
- `shufflenet`: ShuffleNetV2 (mobile-optimized)
- `squeezenet`: SqueezeNet (compact)
- `efficientnet`: EfficientNet (SOTA)

### Federated Learning Setup

```python
"number_of_nodes": 1,                  # Number of server nodes
"number_of_clients_per_node": 6,       # Clients per node
"n_rounds": 10,                        # Training rounds
"min_number_of_clients_in_cluster": 3, # Minimum cluster size
```

### Training Hyperparameters

```python
"batch_size": 32,           # Training batch size
"n_epochs": 5,              # Epochs per round (per client)
"lr": 0.001,                # Learning rate
"weight_decay": 1e-4,       # L2 regularization
"patience": 3,              # Early stopping patience
```

**Optimizers & Schedulers:**
```python
"choice_optimizer": "Adam", # Adam, SGD, AdamW
"choice_scheduler": "StepLR", # StepLR, CosineAnnealing, None
"step_size": 5,             # Scheduler step size
"gamma": 0.1,               # Learning rate decay factor
```

## Privacy Mechanisms

### Differential Privacy

```python
"diff_privacy": True,       # Enable DP
"noise_multiplier": 0.1,    # DP noise multiplier (lower = less noise)
"max_grad_norm": 0.5,       # Gradient clipping threshold
"delta": 1e-5,              # DP delta parameter
"epsilon": 5.0,             # Privacy budget (lower = more private)
```

**Privacy Budget:**
- ε < 1.0: Strong privacy (but lower accuracy)
- ε = 1.0-10: Moderate privacy
- ε > 10: Weak privacy

### Cluster Shuffling

```python
"clustering": True,         # Enable cluster shuffling
"min_number_of_clients_in_cluster": 3,  # Min cluster size
```

Dynamically reorganizes clients into clusters each round to prevent long-term inference.

### SMPC (Secure Multi-Party Computation)

```python
"type_ss": "additif",       # Secret sharing type: "additif" or "shamir"
"threshold": 3,             # Threshold for Shamir's scheme
"m": 3,                     # Number of shares
"ts": 5,                    # Total shares
"use_pytorch_smpc": True,   # Use PyTorch-based SMPC
```

**Secret Sharing Types:**
- **additif**: Simple additive sharing (fast, efficient)
- **shamir**: Shamir's secret sharing (stronger security)

## Gradient Pruning

```python
"gradient_pruning": {
    "enabled": True,                       # Enable gradient pruning
    "keep_ratio": 0.1,                     # Keep 10% of gradients
    "momentum_factor": 0.9,                # Momentum for DGC
    "use_momentum_correction": True,       # Enable momentum correction
    "sample_ratio": 0.01,                  # Sampling for threshold finding
}
```

**Keep Ratio Guidelines:**
- `0.01`: 100x compression, 99% savings (aggressive)
- `0.05`: 20x compression, 95% savings
- `0.1`: 10x compression, 90% savings (recommended)
- `0.2`: 5x compression, 80% savings (conservative)

See [Gradient Pruning Guide](../features/gradient-pruning.md) for details.

## Aggregation Methods

```python
"aggregation": {
    "method": "fedavg",         # Aggregation method
    "krum_malicious": 1,        # Krum: expected malicious clients
    "multi_krum_keep": 3,       # Multi-Krum: clients to keep
    "trim_ratio": 0.2,          # Trimmed Mean: fraction to trim
    "fltrust_root_size": 5000,  # FLTrust: server dataset size
}
```

**Available Methods:**
- **fedavg**: Weighted averaging (baseline, no robustness)
- **krum**: Selects 1 client with lowest distance score
- **multi_krum**: Selects k clients with lowest scores
- **trimmed_mean**: Trims extreme values before averaging
- **median**: Coordinate-wise median
- **fltrust**: Trust-based weighting with server model

See [Aggregation Methods](../features/aggregation-methods.md) for details.

## Poisoning Attacks

```python
"poisoning_attacks": {
    "enabled": False,                       # Enable attacks
    "malicious_clients": [],                # List of malicious client IDs
    "attack_type": "ipm",                   # Attack type
    "attack_intensity": 0.5,                # Attack strength (0.0-1.0)
    "attack_rounds": None,                  # Rounds to attack (None = all)
    "attack_frequency": 1.0,                # Probability of attacking
}
```

**Attack Types:**
- `labelflip`: Label flipping
- `ipm`: Inner Product Manipulation
- `signflip`: Sign flipping
- `noise`: Noise injection
- `alie`: A Little Is Enough
- `backdoor`: Backdoor insertion

Each attack has specific configuration. See [Poisoning Attacks](../attacks/poisoning-attacks.md).

## Attack Evaluation

### Gradient Saving

```python
"save_gradients": True,                 # Save gradients for attack evaluation
"save_gradients_rounds": [1, 2, 3],    # Which rounds to save
"balanced_class_training": True,        # One sample per class (for attacks)
"single_batch_training": True,          # Train on single batch (for attacks)
"max_samples_per_client": 1,            # Limit samples per client
```

**For Attack Evaluation:**
- Enable `save_gradients` to capture gradients
- Use `balanced_class_training` for consistent attack evaluation
- Set `single_batch_training` for faster attack testing

## Output & Logging

```python
"save_figure": True,                    # Save training plots
"matrix_path": "results/CFL/matrix_path",  # Confusion matrix path
"roc_path": "results/CFL/roc_path",     # ROC curve path
```

Results are saved to `results/[experiment_name]/`.

## Advanced Settings

### Model Management

```python
"aggregation_method": "weights",  # "weights" or "gradients"
"use_pytorch_smpc": True,         # Use PyTorch tensors for SMPC
```

### Client Selection

```python
"check_usefulness": False,        # Check client usefulness
"coef_useful": 1.05,              # Usefulness coefficient
"tolerance_ceil": 0.08,           # Usefulness tolerance
```

## Configuration Presets

### Fast Testing

```python
# Quick 1-round test
"n_rounds": 1,
"n_epochs": 1,
"batch_size": 16,
"number_of_clients_per_node": 3,
```

### Privacy Mode

```python
# Maximum privacy protection
"diff_privacy": True,
"clustering": True,
"type_ss": "shamir",
"gradient_pruning": {"enabled": True, "keep_ratio": 0.05},
```

### Attack Evaluation

```python
# Setup for gradient inversion attacks
"save_gradients": True,
"save_gradients_rounds": [1, 2, 3],
"balanced_class_training": True,
"single_batch_training": True,
"batch_size": 1,
```

### Production Training

```python
# Optimized for accuracy
"n_rounds": 50,
"n_epochs": 10,
"batch_size": 64,
"lr": 0.01,
"gradient_pruning": {"enabled": True, "keep_ratio": 0.1},
```

## Configuration Validation

The system validates config on startup. Common errors:

**Error**: `Invalid dataset name`
**Fix**: Use "cifar10", "cifar100", "ffhq128", or "caltech256"

**Error**: `num_classes mismatch`
**Fix**: Set num_classes to match dataset (10 for CIFAR-10, 6 for FFHQ, etc.)

**Error**: `Invalid aggregation method`
**Fix**: Use "fedavg", "krum", "multi_krum", "trimmed_mean", "median", or "fltrust"

## Tips & Best Practices

### For Development
```python
"n_rounds": 3,              # Fast iteration
"batch_size": 32,
"save_gradients": False,    # Save disk space
```

### For Research Papers
```python
"n_rounds": 50,             # Proper convergence
"batch_size": 64,           # Standard size
"save_gradients": True,     # For attack evaluation
# Run multiple experiments with different settings
```

### For Privacy Evaluation
```python
# Test each defense individually, then combined
# Baseline → DP → Clustering → SMPC → All combined
```

### For Attack Resistance
```python
# Test with increasing attack intensity
"attack_intensity": [0.1, 0.3, 0.5, 0.7, 0.9]
# Compare different aggregation methods
```

## Next Steps

- [Quickstart Guide](quickstart.md) - Run experiments
- [Gradient Pruning](../features/gradient-pruning.md) - Communication efficiency
- [Poisoning Attacks](../attacks/poisoning-attacks.md) - Attack evaluation
- [Privacy Defenses](../features/privacy-defenses.md) - SMPC, DP, Clustering
