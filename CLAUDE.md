# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick References

For specific deployment tasks, check these documentation files:

- **vastai-deployment**: Available in `docs/vastai-deployment.md` - Deploy applications on high-performance GPU instances using Vast.AI
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

# Run GIFD (Gradient Inversion from Federated Data) attack with generative priors
# GIFD uses StyleGAN2/BigGAN for high-quality reconstructions via feature-domain optimization
python3 run_gifd.py --gan stylegan2                # Use StyleGAN2 as generative prior
python3 run_gifd.py --gan biggan                   # Use BigGAN for ImageNet-like data
python3 run_gifd.py --gan stylegan2-ada            # Use StyleGAN2-ADA with pre-trained pkl models

# GIFD with specific experiment targeting
python3 run_gifd.py --experiment cifar10_classic_c3_r3 --gan stylegan2
python3 run_gifd.py --rounds 1 2 --clients c0_1 c0_2 --gan biggan

# GIFD attack configurations (optimization intensity)
python3 run_gifd.py --config quick_test    # 1 restart, 1K iterations
python3 run_gifd.py --config default       # 2 restarts, 2K iterations  
python3 run_gifd.py --config aggressive    # 4 restarts, 4K iterations

# Setup pre-trained GAN models for GIFD (downloads StyleGAN2/BigGAN weights)
python3 setup_gan_models.py

# Run federated learning with poisoning attacks
# Configure attacks in config.py first, then run:
python3 main.py

# Test different attack scenarios
python3 test_attacks.py  # Comprehensive attack testing script
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

### Poisoning Attack Configuration
Configure poisoning attacks via the `poisoning_attacks` section in `config.py`:

```python
"poisoning_attacks": {
    "enabled": True,                            # Enable/disable poisoning attacks
    "malicious_clients": ["c0_1", "c0_2"],     # List of malicious client IDs
    "attack_type": "labelflip",                 # Attack type (see below)
    "attack_intensity": 0.2,                   # Attack strength (0.0 to 1.0)
    "attack_rounds": None,                     # Specific rounds to attack (None = all)
    "attack_frequency": 1.0,                   # Probability of attacking each round
    
    # Attack-specific configurations (see individual attack sections)
}
```

**Available Attack Types**:
- `labelflip`: Label flipping attacks on training data
- `ipm`: Inner Product Manipulation on gradients
- `signflip`: Sign flipping on gradients  
- `noise`: Noise injection into gradients
- `alie`: A Little Is Enough byzantine attack
- `backdoor`: Backdoor trigger insertion

**Label Flipping Configuration**:
```python
"labelflip_config": {
    "flip_type": "targeted",               # targeted, random, all_to_one
    "source_class": None,                  # Class to flip from (None = all)
    "target_class": 0,                     # Target class for flipping
    "num_classes": 10                      # Number of classes in dataset
}
```

**IPM (Inner Product Manipulation) Configuration**:
```python
"ipm_config": {
    "target_level": "client",              # client, cross_cluster
    "lambda_param": 0.2,                   # Manipulation strength
}
```

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
   - RSA key management for encrypted communication

4. **models/** - Neural network architectures in `architectures/`:
   - ConvNet (vulnerable to gradient attacks), SimpleNet, MobileNet, ResNet, ShuffleNet, SqueezeNet, EfficientNet
   - Factory pattern for model instantiation
   - Support for pretrained models

5. **data/** - Dataset management:
   - `loaders.py`: CIFAR-10/100, Caltech256, FFHQ dataset loading
   - `ffhq_dataset.py`: Custom FFHQ dataset implementation  
   - Dirichlet distribution for non-IID data partitioning

6. **attacks/poisoning/** - Modular Poisoning Attack Framework:
   - `base_poisoning_attack.py`: Abstract base class for all poisoning attacks
   - `attack_factory.py`: Factory pattern for dynamic attack instantiation
   - `labelflip_attack.py`: Label flipping attacks (targeted, random, all-to-one)
   - `ipm_attack.py`: Inner Product Manipulation with cross-cluster targeting
   - `signflip_attack.py`: Sign flipping attacks on gradients
   - `noise_attack.py`: Gaussian/uniform/Laplacian noise injection
   - `alie_attack.py`: A Little Is Enough byzantine attack
   - `backdoor_attack.py`: Backdoor trigger insertion attacks
   - `evaluation.py`: Attack effectiveness metrics and analysis

7. **Privacy Attack Evaluation Framework**:
   - `run_grad_inv.py`: Unified gradient inversion attack runner with intelligent experiment discovery
   - `run_gifd.py`: GIFD (Gradient Inversion from Federated Data) attack using generative models
   - `attacks/`: Modular attack framework with configurable attack intensities
     - `gradient_inversion.py`: GradientInversionAttacker class with multiple configurations
     - `attack_configs.py`: Predefined attack scenarios (quick_test, default, aggressive, etc.)
     - `utils/`: Attack utilities for data loading, metrics, and visualization
   - `mia_attack.py`: Membership inference using shadow models
   - `GIFD_Gradient_Inversion_Attack/`: GIFD implementation with StyleGAN2 and BigGAN support
     - Advanced gradient inversion using generative priors
     - Multiple GAN architectures for improved reconstruction quality
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

### Poisoning Attack Defense Evaluation

The framework includes comprehensive poisoning attack capabilities to evaluate defense mechanisms:

**Attack Categories**:
1. **Data-level Attacks**: 
   - Label flipping (targeted/random/all-to-one)
   - Backdoor trigger insertion
   
2. **Gradient-level Attacks**:
   - Inner Product Manipulation (IPM) with cross-cluster targeting
   - Sign flipping attacks
   - Gaussian/uniform/Laplacian noise injection
   - A Little Is Enough (ALIE) byzantine attack

**Defense Evaluation Workflow**:
1. **Baseline Testing**: Run attacks without defense mechanisms
2. **Individual Defense Testing**: Test each defense (clustering, SMPC, DP) individually  
3. **Combined Defense Testing**: Evaluate layered defense effectiveness
4. **Attack Intensity Scaling**: Test with different attack intensities (0.1-0.8)
5. **Cross-cluster Attack Evaluation**: Test sophisticated attacks targeting cluster shuffling

**Attack Effectiveness Metrics**:
- Training accuracy degradation on malicious clients
- Global model accuracy impact  
- Attack detection through gradient analysis
- Defense mechanism bypass success rate

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