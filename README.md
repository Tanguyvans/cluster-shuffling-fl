# Cluster Shuffling Federated Learning

A privacy-preserving federated learning system that implements cluster shuffling and secure multi-party computation (SMPC) to protect against various privacy attacks while maintaining model performance.

## Overview

This project implements a federated learning framework with the following key privacy-preserving features:

- **Cluster Shuffling**: Dynamically reorganizes clients into different clusters across training rounds to prevent inference attacks
- **Secure Multi-Party Computation (SMPC)**: Uses secret sharing schemes (additive and Shamir's) to protect model updates
- **Differential Privacy**: Adds calibrated noise to model parameters to provide formal privacy guarantees
- **Privacy Attack Evaluation**: Includes implementations of Membership Inference Attacks (MIA) and Gradient Inversion Attacks

## Architecture

The system consists of:

- **Nodes**: Coordinate federated learning rounds and manage clusters
- **Clients**: Train local models and participate in secure aggregation
- **Security Layer**: Implements SMPC protocols and differential privacy mechanisms
- **Attack Modules**: Evaluate privacy vulnerabilities through various attack scenarios

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Tanguyvans/cluster-shuffling-fl.git
cd cluster-shuffling-fl
```

2. Install dependencies:

```bash
pip3 install -r requirements.txt
```

3. Download CIFAR-10 dataset (automatically handled on first run)

## Configuration

Edit `config.py` to customize the federated learning setup:

```python
settings = {
    "name_dataset": "cifar10",           # Dataset: cifar10, cifar100, caltech256
    "arch": "simplenet",                 # Model: simplenet, mobilenet, resnet18, shufflenet
    "number_of_clients_per_node": 6,     # Number of clients per node
    "n_rounds": 10,                      # Number of federated rounds
    "lr": 0.001,                         # Learning rate
    "batch_size": 32,                    # Batch size

    # Privacy settings
    "diff_privacy": False,               # Enable differential privacy
    "noise_multiplier": 0.1,             # DP noise multiplier
    "epsilon": 5.0,                      # DP privacy budget

    # Clustering and SMPC
    "clustering": True,                  # Enable cluster shuffling
    "type_ss": "additif",               # Secret sharing: additif, shamir
    "threshold": 3,                      # SMPC threshold
    "min_number_of_clients_in_cluster": 3,
}
```

## Running the System

### Basic Federated Learning

Run the main federated learning training:

```bash
python3 main.py
```

This will:

1. Initialize the federated learning environment
2. Create clients and distribute data
3. Train models for the specified number of rounds
4. Apply privacy mechanisms (clustering, SMPC, DP) as configured
5. Save trained models and metrics

### Monitoring Training

Training progress is logged to `results/CFL/output.txt` and includes:

- Per-client training/validation/test metrics
- Communication overhead measurements
- Privacy budget consumption (if DP enabled)
- Cluster assignments and shuffling events

### Trained Models

Models are managed through the centralized ModelManager with structured organization:

```
results/[experiment_name]/
├── models/
│   ├── clients/round_XXX/        # Individual client models per round
│   ├── global/                   # Aggregated global models
│   ├── clusters/round_XXX/       # Cluster-level aggregations
│   └── fragments/round_XXX/      # SMPC secret shares
├── logs/                         # Training and experiment logs
├── metrics/                      # Energy, communication, time tracking
└── config.json                   # Experiment configuration
```

## Privacy Attack Evaluation

The system includes implementations of state-of-the-art privacy attacks to evaluate the effectiveness of the privacy-preserving mechanisms.

### MIA Attack

**Membership Inference Attack** attempts to determine if a specific data sample was used in training a target model.

#### How it works:

1. **Shadow Model Training**: Trains multiple shadow models on datasets with known membership
2. **Attack Model Training**: Uses shadow model predictions to train a binary classifier
3. **Inference**: Applies the attack model to determine membership of target samples

#### Running MIA Attack:

```bash
python3 mia_attack.py
```

**Configuration**:

- `NUM_SHADOWS = 3`: Number of shadow models to train
- `SHADOW_DATASET_SIZE = 4000`: Size of shadow training datasets
- `ATTACK_TEST_DATASET_SIZE = 4000`: Size of attack evaluation dataset

**Attack Results**:

- Attack accuracy ~50% indicates strong privacy protection (random guessing)
- Attack accuracy >60% suggests potential privacy vulnerabilities
- Results are saved with detailed metrics and analysis

**Key Features**:

- Supports multiple model architectures (SimpleNet, ResNet, MobileNet)
- Automatic conversion between PyTorch and TensorFlow models
- Comprehensive evaluation metrics (accuracy, precision, recall, AUC)

### Inversion Attack

**Gradient Inversion Attack** attempts to reconstruct training data from model gradients shared during federated learning.

#### How it works:

1. **Gradient Extraction**: Captures gradients from a trained model on target data
2. **Dummy Data Initialization**: Creates random dummy images and labels
3. **Gradient Matching**: Optimizes dummy data to match target gradients
4. **Reconstruction**: Outputs reconstructed images that approximate original data

#### Running Inversion Attack:

```bash
cd breaching_attack
python3 simple_gradient_attack.py
```

**Configuration**:

- `num_iterations = 5000`: Number of optimization iterations
- `lr = 0.1`: Learning rate for dummy data optimization
- `batch_size = 32`: Batch size for reconstruction

**Attack Parameters**:

- **Total Variation Loss**: Promotes image smoothness
- **L2 Regularization**: Prevents overfitting to noise
- **Gradient Matching Loss**: Core objective function

**Results Visualization**:

- Side-by-side comparison of original vs reconstructed images
- Quantitative metrics: MSE, PSNR, SSIM
- Attack success rates across different model states

**Key Features**:

- Supports multiple trained model formats
- Comprehensive evaluation metrics
- Visualization of reconstruction quality
- Batch processing for efficiency

## Privacy Protection Mechanisms

### Cluster Shuffling

- Dynamically reassigns clients to different clusters each round
- Prevents long-term inference attacks based on cluster membership
- Maintains model performance while reducing attack surface

### Secure Multi-Party Computation (SMPC)

- **Additive Secret Sharing**: Simple and efficient for basic privacy
- **Shamir's Secret Sharing**: Provides stronger security guarantees
- Protects individual model updates during aggregation

### Differential Privacy

- Adds calibrated noise to model parameters
- Provides formal privacy guarantees with (ε, δ)-DP
- Configurable privacy budget management

## Experimental Results

### Privacy vs Utility Trade-offs

- **No Privacy**: High utility, vulnerable to attacks
- **Clustering Only**: Moderate privacy, minimal utility loss
- **SMPC + Clustering**: Strong privacy, <5% utility degradation
- **Full Protection**: Maximum privacy, 10-15% utility cost

### Attack Resistance

- **MIA Success Rate**: Reduced from 85% to 52% with full protection
- **Inversion Quality**: PSNR reduced from 25dB to 8dB
- **Communication Overhead**: 2-3x increase with SMPC

## Project Structure

```
cluster-shuffling-fl/
├── main.py                          # Main federated learning orchestrator
├── config.py                        # Centralized configuration settings
├── ATTACK_GUIDE.md                  # Comprehensive attack usage guide
├── CLAUDE.md                        # Project documentation for Claude Code
│
├── federated/                       # Federated learning implementation
│   ├── client.py                   # Client class for local training
│   ├── server.py                   # Node class for coordination
│   ├── aggregation.py              # Aggregation strategies
│   ├── training.py                 # Training and evaluation logic
│   ├── flower_client.py            # Flower framework wrapper
│   └── factory.py                  # Factory functions for nodes/clients
│
├── models/                          # Neural network architectures
│   ├── factory.py                  # Model factory pattern
│   └── architectures/              # Model implementations
│       ├── simple_net.py           # Lightweight CNN
│       ├── conv_net.py             # Vulnerable ConvNet for testing
│       ├── mobilenet.py            # MobileNet architecture
│       ├── resnet.py               # ResNet variants
│       ├── shuffle_net.py          # ShuffleNet architecture
│       ├── squeeze_net.py          # SqueezeNet architecture
│       └── efficient_net.py        # EfficientNet architecture
│
├── data/                            # Dataset management
│   ├── loaders.py                  # CIFAR-10/100, Caltech256 loaders
│   └── ffhq_dataset.py             # FFHQ face dataset loader
│
├── security/                        # Privacy-preserving mechanisms
│   └── secret_sharing.py           # Additive and Shamir secret sharing
│
├── attacks/                         # Attack framework
│   ├── poisoning/                  # Poisoning attacks
│   │   ├── base_poisoning_attack.py      # Abstract base class
│   │   ├── attack_factory.py             # Attack factory pattern
│   │   ├── labelflip_attack.py           # Label flipping attacks
│   │   ├── ipm_attack.py                 # Inner Product Manipulation
│   │   ├── signflip_attack.py            # Sign flipping attacks
│   │   ├── noise_attack.py               # Noise injection attacks
│   │   ├── alie_attack.py                # A Little Is Enough attack
│   │   ├── backdoor_attack.py            # Backdoor attacks
│   │   └── evaluation.py                 # Attack metrics
│   └── utils/                      # Attack utilities
│       ├── data_loader.py          # Attack data loading
│       ├── metrics.py              # Attack evaluation metrics
│       └── visualization.py        # Attack visualization
│
├── utils/                           # Utility modules
│   ├── model_manager.py            # Centralized model/gradient management
│   ├── model_metadata.py           # Model metadata handling
│   ├── model_paths.py              # Path management
│   ├── system_metrics.py           # Energy, communication, time tracking
│   ├── metrics.py                  # Training metrics
│   ├── visualization.py            # Visualization utilities
│   ├── device.py                   # Device management
│   └── optimization.py             # Optimization utilities
│
├── core/                            # Core training logic
│   └── training.py                 # Training orchestration
│
├── mia_attack.py                    # Membership inference attack
├── run_inference_attack.py          # Gradient inversion attack runner
├── attack_fl_ffhq.py                # FFHQ-specific attack script
├── attack_ffhq.py                   # Alternative FFHQ attack implementation
├── train_ffhq_resnet.py             # FFHQ ResNet training
│
├── results/                         # Training results and metrics
│   └── [experiment_name]/
│       ├── models/                 # Saved models per round
│       ├── logs/                   # Training logs
│       ├── metrics/                # System metrics
│       └── config.json             # Experiment configuration
│
├── keys/                            # RSA keys for SMPC
└── requirements.txt                 # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request
