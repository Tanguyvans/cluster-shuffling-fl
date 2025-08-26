# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the System
```bash
# Install dependencies
pip3 install -r requirements.txt

# Run federated learning training
python3 main.py

# Run membership inference attack evaluation
python3 mia_attack.py

# Run gradient inversion attack evaluation
python3 run_grad_inv.py

# List available experiments for attack
python3 run_grad_inv.py --list

# Advanced gradient inversion attack targeting
python3 run_grad_inv.py --config aggressive --rounds 1 2 --clients c0_1 c0_2
```

### Configuration
All settings are centralized in `config.py`. Modify this file rather than using command-line arguments. Key settings:
- `name_dataset`: "cifar10", "cifar100", "ffhq128", or "caltech256" 
- `arch`: "convnet", "simplenet", "mobilenet", "resnet18", "shufflenet", "squeezenet", or "efficientnet"
- `clustering`: True/False (enable cluster shuffling)
- `diff_privacy`: True/False (enable differential privacy)
- `type_ss`: "additif" or "shamir" (secret sharing type)
- `n_rounds`: Number of federated learning rounds
- `number_of_clients_per_node`: Clients per node (default 6)
- `batch_size`: Training batch size (default 10)
- `balanced_class_training`: True/False (ensure one sample per class for vulnerability testing)
- `save_gradients`: True/False (enable gradient saving for attack evaluation)
- `aggregation_method`: "weights" or "gradients" (SMPC aggregation method)

## Architecture

This is a privacy-preserving federated learning framework implementing cluster shuffling with secure multi-party computation (SMPC). The system evaluates defense mechanisms against membership inference and gradient inversion attacks.

### Core Components and Flow

1. **main.py** - Central orchestrator that:
   - Initializes nodes and clients with distributed datasets
   - Manages federated learning rounds with optional clustering
   - Applies SMPC for secure aggregation
   - Tracks metrics (energy, communication, time)

2. **federated/** - Federated learning implementation:
   - `client.py`: Client class managing local training
   - `server.py`: Node class coordinating aggregation  
   - `flower_client.py`: Flower framework wrapper
   - `factory.py`: Factory functions for creating nodes/clients
   - `training.py`: Training and evaluation logic

3. **security/** - Privacy-preserving mechanisms:
   - `secret_sharing.py`: Additive and Shamir secret sharing implementations
   - `attacks.py`: Data poisoning and attack utilities
   - RSA key management for encrypted communication

4. **models/** - Neural network architectures in `architectures/`:
   - ConvNet (vulnerable to gradient attacks), SimpleNet, MobileNet, ResNet, ShuffleNet, SqueezeNet, EfficientNet
   - Factory pattern for model instantiation
   - Support for pretrained models

5. **data/** - Dataset management:
   - `loaders.py`: CIFAR-10/100, Caltech256, FFHQ dataset loading
   - `ffhq_dataset.py`: Custom FFHQ dataset implementation  
   - Dirichlet distribution for non-IID data partitioning

6. **Privacy Attack Evaluation Framework**:
   - `run_grad_inv.py`: Unified gradient inversion attack runner with intelligent experiment discovery
   - `attacks/`: Modular attack framework with configurable attack intensities
     - `gradient_inversion.py`: GradientInversionAttacker class with multiple configurations
     - `attack_configs.py`: Predefined attack scenarios (quick_test, default, aggressive, etc.)
     - `utils/`: Attack utilities for data loading, metrics, and visualization
   - `mia_attack.py`: Membership inference using shadow models
   - `inversefed/`: InverseFed library integration for advanced attacks

### Key Implementation Details

- **Model Management**: Centralized through `utils/model_manager.py` with clean directory structure:

  ```text
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

- **ModelManager Features**:
  - Structured model saving with consistent metadata
  - Automatic directory creation and management
  - Support for client, global, cluster, and fragment models
  - Gradient saving for attack evaluation (enabled via `save_gradients` config)
  - Centralized experiment tracking
  - Balanced class training support for enhanced vulnerability testing

- **Metrics Tracking**: `utils/system_metrics.py` provides:
  - Energy consumption via pyRAPL/pynvml
  - Communication overhead measurement
  - Training time tracking per round/client

- **SMPC Implementation**:
  - RSA keys auto-generated in `keys/` on first run
  - Supports threshold-based reconstruction
  - Configurable share count and cluster sizes
  - Supports both weight-based and gradient-based aggregation methods

- **Differential Privacy**: 
  - Opacus integration for gradient clipping and noise addition
  - Configurable epsilon, delta, noise multiplier
  - Privacy budget tracking across rounds

### Data Flow

```text
Clients → Local Training → SMPC Encoding → Cluster Aggregation → Node Aggregation → SMPC Decoding → Global Model
    ↑                                                                                                      ↓
    └──────────────────────────────── Broadcast Updated Global Model ──────────────────────────────────────┘
```

### Privacy Mechanisms

- **Cluster Shuffling**: Dynamic client-cluster reassignment each round prevents long-term inference
- **SMPC**: Secret sharing prevents individual gradient exposure during aggregation
- **Differential Privacy**: Calibrated noise addition provides formal privacy guarantees
- **Combined Defense**: All mechanisms can run simultaneously for maximum protection

### Attack Evaluation Workflow

The framework provides systematic privacy evaluation through gradient inversion attacks:

1. **Training with Gradient Capture**:
   ```bash
   # Enable gradient saving in config.py
   "save_gradients": True,
   "balanced_class_training": True,
   "save_gradients_rounds": [1, 2, 3]
   
   # Run training
   python3 main.py
   ```

2. **Attack Evaluation**:
   ```bash
   # Quick vulnerability check
   python3 run_grad_inv.py --config quick_test
   
   # Comprehensive attack
   python3 run_grad_inv.py --config aggressive
   
   # Compare attack intensities
   python3 run_grad_inv.py --compare default aggressive high_quality
   ```

3. **Privacy Assessment**:
   - **PSNR > 25 dB**: Vulnerable (high-quality reconstruction)
   - **PSNR 20-25 dB**: Weak protection
   - **PSNR 15-20 dB**: Moderate protection  
   - **PSNR < 15 dB**: Strong protection

4. **Systematic Evaluation**:
   - Test baseline (no privacy mechanisms)
   - Evaluate SMPC-only protection
   - Test differential privacy effectiveness
   - Compare combined defense mechanisms

### Attack Configurations

Available attack intensities for systematic evaluation:

- `quick_test`: Fast testing (1 restart, 8K iterations)
- `default`: Balanced evaluation (2 restarts, 24K iterations)  
- `aggressive`: Strong attack (5 restarts, 48K iterations)
- `conservative`: Light testing (1 restart, 12K iterations)
- `high_quality`: Maximum reconstruction (8 restarts, 60K iterations)

Use different configurations to evaluate privacy protection under varying attack intensities.