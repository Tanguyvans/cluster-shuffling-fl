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

### Core Flow
1. **main.py** orchestrates FL rounds, managing the Node (server) and Clients
2. **flowerclient.py** wraps the Flower framework for FL communication
3. **going_modular/** contains core logic:
   - `model.py`: Neural network architectures
   - `engine.py`: Training and evaluation loops
   - `security.py`: SMPC implementations (additive and Shamir secret sharing)
   - `data_setup.py`: Dataset loading and partitioning

### Key Design Patterns
- **Flower Framework Integration**: Uses Flower's client-server architecture with custom strategies
- **Modular Security**: Privacy mechanisms (SMPC, DP, clustering) are toggleable via config
- **Attack Evaluation**: Separate scripts test defense effectiveness post-training
- **Metrics Collection**: Automatic tracking of energy consumption and communication costs

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