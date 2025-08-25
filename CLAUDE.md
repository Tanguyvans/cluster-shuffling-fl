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

# Run gradient inversion attack
python3 gradient_inversion_attack.py

# Run single image training and attack test
python3 train_and_attack_single_image.py
```

### Configuration
All settings are centralized in `config.py`. Modify this file rather than using command-line arguments. Key settings:
- `name_dataset`: "cifar10", "cifar100", "ffhq128", or "caltech256" 
- `arch`: "simplenet", "mobilenet", "resnet18", "shufflenet", "squeezenet", or "efficientnet"
- `clustering`: True/False (enable cluster shuffling)
- `diff_privacy`: True/False (enable differential privacy)
- `type_ss`: "additif" or "shamir" (secret sharing type)
- `n_rounds`: Number of federated learning rounds
- `number_of_clients_per_node`: Clients per node (default 6)
- `batch_size`: Training batch size (default 32)

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
   - SimpleNet, MobileNet, ResNet, ShuffleNet, SqueezeNet, EfficientNet
   - Factory pattern for model instantiation
   - Support for pretrained models

5. **data/** - Dataset management:
   - `loaders.py`: CIFAR-10/100, Caltech256, FFHQ dataset loading
   - `ffhq_dataset.py`: Custom FFHQ dataset implementation  
   - Dirichlet distribution for non-IID data partitioning

6. **Attack Evaluation Scripts**:
   - `mia_attack.py`: Membership inference using shadow models
   - `gradient_inversion_attack.py`: Smart gradient inversion prioritizing vulnerable rounds
   - `inverting_attack.py`: Basic gradient inversion implementation
   - `inversefed/`: InverseFed library integration for advanced attacks

### Key Implementation Details

- **Results Structure**: `results/CFL/` contains:
  - `global_models/`: Aggregated models per round
  - `client_models/`: Individual client models and round checkpoints
  - `cluster_models/`: Cluster-level aggregations
  - `fragments/`: SMPC secret shares
  - Metrics files: energy, communication, time tracking

- **Metrics Tracking**: `utils/system_metrics.py` provides:
  - Energy consumption via pyRAPL/pynvml
  - Communication overhead measurement
  - Training time tracking per round/client

- **SMPC Implementation**:
  - RSA keys auto-generated in `keys/` on first run
  - Supports threshold-based reconstruction
  - Configurable share count and cluster sizes

- **Differential Privacy**: 
  - Opacus integration for gradient clipping and noise addition
  - Configurable epsilon, delta, noise multiplier
  - Privacy budget tracking across rounds

### Data Flow
```
Clients → Local Training → SMPC Encoding → Cluster Aggregation → Node Aggregation → SMPC Decoding → Global Model
    ↑                                                                                                      ↓
    └──────────────────────────────── Broadcast Updated Global Model ──────────────────────────────────────┘
```

### Privacy Mechanisms
- **Cluster Shuffling**: Dynamic client-cluster reassignment each round prevents long-term inference
- **SMPC**: Secret sharing prevents individual gradient exposure during aggregation
- **Differential Privacy**: Calibrated noise addition provides formal privacy guarantees
- **Combined Defense**: All mechanisms can run simultaneously for maximum protection