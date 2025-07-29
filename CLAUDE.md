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
python3 breaching_attack/simple_gradient_attack.py

# Run DeepInversion attack
cd DeepInversion && python3 training_DeepInversion.py
```

### Configuration
All settings are centralized in `config.py`. Modify this file rather than using command-line arguments. Key settings:
- `name_dataset`: "cifar10", "cifar100", or "caltech256"
- `arch`: "simplenet", "mobilenet", "resnet18", or "shufflenet"
- `clustering`: True/False (enable cluster shuffling)
- `diff_privacy`: True/False (enable differential privacy)
- `type_ss`: "additif" or "shamir" (secret sharing type)

## Architecture

This is a privacy-preserving federated learning framework implementing cluster shuffling with secure multi-party computation (SMPC). The system evaluates defense mechanisms against membership inference and gradient inversion attacks.

### New Modular Structure
The codebase has been restructured for better organization:

```
├── core/                    # Core ML components
│   ├── models.py           # Neural network architectures (SimpleNet, ResNet18, etc.)
│   └── training.py         # Training/evaluation loops with early stopping
├── data/                   # Data handling
│   └── loaders.py          # Dataset loading, partitioning, normalization
├── security/               # Privacy and security mechanisms  
│   ├── secret_sharing.py   # SMPC implementations (additive & Shamir)
│   └── attacks.py          # Data poisoning functions
├── federated/             # Federated learning components
│   ├── client.py          # Client class with SMPC capabilities
│   ├── server.py          # Node class for aggregation and coordination
│   ├── flower_client.py   # Flower framework integration
│   ├── factory.py         # Client/Node creation functions
│   └── training.py        # Training orchestration logic
├── utils/                 # Utilities split by functionality
│   ├── config.py          # Parameter initialization 
│   ├── device.py          # Device selection (CPU/GPU/MPS)
│   ├── metrics.py         # Performance metrics (sMAPE)
│   ├── optimization.py    # Loss functions, optimizers, schedulers
│   ├── visualization.py   # Plotting, ROC curves, confusion matrices
│   └── model_utils.py     # Model parameter handling
├── main.py                # Main orchestration script
└── config.py              # Central configuration
```

### Core Flow
1. **main.py** orchestrates FL rounds using factory functions from `federated/`
2. **federated/flower_client.py** wraps the Flower framework for FL communication
3. **federated/client.py** and **federated/server.py** handle the core FL logic
4. **security/** modules provide SMPC and attack capabilities
5. **core/** modules contain the ML pipeline components

### Key Design Patterns
- **Flower Framework Integration**: Uses Flower's client-server architecture with custom strategies
- **Modular Security**: Privacy mechanisms (SMPC, DP, clustering) are toggleable via config
- **Attack Evaluation**: Separate scripts test defense effectiveness post-training
- **Metrics Collection**: Automatic tracking of energy consumption and communication costs
- **Factory Pattern**: `federated/factory.py` creates clients and nodes with proper configuration

### Important Implementation Details
- Results stored in `results/CFL/` with subdirectories for models, metrics, and logs
- Client models saved individually for attack evaluation
- RSA keys for SMPC auto-generated in `keys/` directory on first run
- Supports both CPU and GPU execution (auto-detected)
- Dataset partitioning uses Dirichlet distribution for non-IID data

### Data Flow
```
Clients → Local Training → SMPC Encoding → Node Aggregation → SMPC Decoding → Global Model Update
    ↑                                                                                    ↓
    └────────────────────────── Broadcast New Model ────────────────────────────────────┘
```

### Privacy Mechanisms
- **Cluster Shuffling**: Clients dynamically reassigned to clusters each round
- **SMPC**: Model updates secret-shared among clients before aggregation  
- **Differential Privacy**: Opacus library adds calibrated noise to gradients
- **Combined Defense**: All mechanisms can be enabled simultaneously for maximum privacy