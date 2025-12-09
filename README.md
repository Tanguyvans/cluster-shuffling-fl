# Cluster Shuffling Federated Learning

A privacy-preserving federated learning system with **cluster shuffling**, **SMPC**, and **gradient pruning** for communication-efficient, secure distributed training.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/Tanguyvans/cluster-shuffling-fl.git
cd cluster-shuffling-fl
pip3 install -r requirements.txt

# Run
python3 main.py
```

**Result**: Federated learning on CIFAR-10 with 6 clients, 10 rounds, **80% communication savings** from gradient pruning!

📖 **New to FL?** → [Quickstart Guide](docs/getting-started/quickstart.md)

---

## ✨ Key Features

### Privacy & Security
- **🔄 Cluster Shuffling**: Dynamic client reorganization prevents long-term inference
- **🔐 SMPC**: Secret sharing (additive & Shamir's) protects model updates
- **🛡️ Differential Privacy**: Calibrated noise for formal privacy guarantees

### Communication Efficiency
- **📉 Gradient Pruning** (NEW!): 80% communication reduction via Deep Gradient Compression (DGC)
- **⚡ Top-k Sparsification**: Send only 10% of gradients with momentum correction
- **🔗 Compatible**: Works with SMPC, DP, and all privacy mechanisms

### Attack Evaluation
- **⚔️ Poisoning Attacks**: 6 attack types (Label Flip, IPM, ALIE, Backdoor, etc.)
- **🔍 Privacy Attacks**: Gradient inversion, membership inference
- **📊 Comprehensive Metrics**: PSNR, accuracy, communication overhead

### Byzantine Robustness
- **Krum**, Multi-Krum
- **Trimmed Mean**, Median
- **FLTrust** - Trust-based aggregation

---

## 📚 Documentation

### Getting Started
- [Installation Guide](docs/getting-started/installation.md) - Setup & dependencies
- [Quickstart (5 min)](docs/getting-started/quickstart.md) - First FL experiment
- [Configuration](docs/getting-started/configuration.md) - Complete config.py reference

### Core Features
- [Gradient Pruning](docs/features/gradient-pruning.md) - 80% communication savings
- [Privacy Defenses](docs/features/privacy-defenses.md) - SMPC, DP, Clustering
- [Aggregation Methods](docs/features/aggregation-methods.md) - Krum, FLTrust, etc.

### Attack Evaluation
- [Poisoning Attacks](docs/attacks/poisoning-attacks.md) - 6 attack types
- [Gradient Inversion](docs/attacks/gradient-inversion.md) - Privacy attacks
- [Pruned Models](docs/attacks/pruned-models.md) - Attack comparison

### Measurement
- [Communication Metrics](docs/measurement/communication.md) - Measure pruning impact

📖 **[Full Documentation Index](docs/README.md)**

---

## 🎯 Use Cases

### Research & Evaluation

```python
# Test gradient pruning impact
"gradient_pruning": {"enabled": True, "keep_ratio": 0.1}
python3 main.py

# Compare attack resistance
python3 run_grad_inv.py --config aggressive
```

### Privacy Evaluation

```python
# Enable all privacy mechanisms
"diff_privacy": True,
"clustering": True,
"type_ss": "shamir",
"gradient_pruning": {"enabled": True}
```

### Attack Testing

```python
# Test poisoning attacks
"poisoning_attacks": {
    "enabled": True,
    "malicious_clients": ["c0_1"],
    "attack_type": "ipm",
    "attack_intensity": 0.5
}
```

---

## 📊 Results

### Communication Efficiency

| Method | Compression | Savings | Accuracy Impact |
|--------|-------------|---------|-----------------|
| Baseline | 1.0x | 0% | - |
| **Gradient Pruning (k=0.1)** | **5.0x** | **80%** | **<1%** |
| Pruning (k=0.05) | 10.0x | 90% | ~2% |

### Privacy Protection (PSNR - lower is better)

| Defense | Gradient Inversion PSNR | Privacy Level |
|---------|-------------------------|---------------|
| None | 28 dB | ❌ Vulnerable |
| SMPC | 18 dB | ✅ Moderate |
| SMPC + Pruning | 15 dB | ✅ Strong |
| SMPC + DP | 12 dB | ✅✅ Very Strong |

### Attack Resistance

| Defense | IPM Attack Impact | Label Flip Impact |
|---------|-------------------|-------------------|
| FedAvg | -40% accuracy | -35% accuracy |
| Krum | -13% accuracy | -8% accuracy |
| Krum + Clustering | -4% accuracy | -2% accuracy |

---

## 🏗️ Architecture

```
┌─────────────┐
│   Clients   │ ──► Local Training
└─────────────┘
       │
       ├──► Gradient Pruning (80% reduction)
       │
       ├──► SMPC Secret Sharing
       │
       ▼
┌─────────────┐
│ Aggregation │ ──► Krum / FedAvg / FLTrust
└─────────────┘
       │
       ▼
┌─────────────┐
│ Global Model│ ──► Broadcast to Clients
└─────────────┘
```

---

## 🔧 Configuration

Edit `config.py` for quick customization:

```python
# Dataset & Model
"name_dataset": "cifar10",      # cifar10, cifar100, ffhq128
"arch": "simplenet",            # simplenet, resnet18, mobilenet

# Federated Learning
"n_rounds": 10,                 # Training rounds
"number_of_clients_per_node": 6,# Clients per node

# Gradient Pruning (NEW!)
"gradient_pruning": {
    "enabled": True,            # 80% communication savings
    "keep_ratio": 0.1,          # Keep 10% of gradients
}

# Privacy
"diff_privacy": True,           # Enable DP
"clustering": True,             # Cluster shuffling

# Aggregation
"aggregation": {
    "method": "krum",           # fedavg, krum, fltrust
}
```

📖 [Complete Configuration Guide](docs/getting-started/configuration.md)

---

## 📁 Project Structure

```
cluster-shuffling-fl/
├── main.py                     # Main FL orchestrator
├── config.py                   # Configuration settings
│
├── docs/                       # 📚 Documentation
│   ├── getting-started/        # Installation, quickstart, config
│   ├── features/               # Gradient pruning, privacy, etc.
│   ├── attacks/                # Poisoning, gradient inversion
│   └── measurement/            # Metrics and evaluation
│
├── federated/                  # FL implementation
│   ├── client.py               # Client training
│   ├── server.py               # Server aggregation
│   └── flower_client.py        # Flower wrapper
│
├── security/                   # Privacy mechanisms
│   ├── secret_sharing.py       # SMPC implementation
│   └── gradient_pruning.py     # DGC implementation
│
├── attacks/poisoning/          # Attack framework
│   ├── labelflip_attack.py
│   ├── ipm_attack.py
│   └── ...
│
└── models/architectures/       # Neural network models
    ├── simplenet.py
    ├── resnet.py
    └── ...
```

---

## 🧪 Testing

```bash
# Test gradient pruning
python3 test_gradient_pruning.py

# Run gradient inversion attack
python3 run_grad_inv.py --config default

# Measure communication savings
python3 measure_communication.py --keep-ratio 0.1
```

---

## 🔓 Gradient Inversion Attacks Setup

The framework supports advanced gradient inversion attacks (GIAS, GIFD) using StyleGAN2 generative priors.

### Quick Setup (GIAS - Recommended)

GIAS works out of the box with the included `Gs.pth` weights:

```bash
# Run GIAS attack on FFHQ
python3 attack_fl_ffhq.py --attack-type gias
```

### Advanced Setup (GIFD)

GIFD requires additional model files for inter-layer optimization:

```bash
# Install gdown for Google Drive downloads
pip install gdown

# 1. Download gaussian_fit.pt (shape predictor) - place in project root
gdown --id 1c1qtz3MVTAvJpYvsMIR5MoSvdiwN2DGb

# 2. Create directory for StyleGAN2 checkpoint
mkdir -p exploitai/attacks/inference/gifd_core/genmodels/stylegan2_io

# 3. Download StyleGAN2 checkpoint (~550MB)
gdown --id 1JCBiKY_yUixTa6F1eflABL88T4cii2GR -O exploitai/attacks/inference/gifd_core/genmodels/stylegan2_io/stylegan2-ffhq-config-f.pt

# 4. Run GIFD attack
python3 attack_fl_ffhq.py --attack-type gifd
```

### Attack Comparison

| Attack | Setup | Quality | Speed | Use Case |
|--------|-------|---------|-------|----------|
| **GIAS** | ✅ Ready | Good | Fast | Quick evaluation |
| **GIFD** | Requires downloads | Best | Slow | Research, publications |

📖 See [Gradient Inversion Guide](docs/attacks/gradient-inversion.md) for detailed usage.

---

## 📖 Research & Papers

This framework implements and evaluates:

- **Deep Gradient Compression** (Lin et al., ICLR 2018)
- **Cluster Shuffling** for federated learning
- **Byzantine-robust aggregation** (Krum, Trimmed Mean)
- **Gradient inversion attacks** (DLG, iDLG, GIAS, GIFD)

See [Research Papers](docs/reference/papers.md) for full citations.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional attack implementations
- More aggregation methods
- Enhanced privacy mechanisms
- Documentation improvements

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Flower](https://flower.dev/) - Federated learning framework
- [Opacus](https://opacus.ai/) - Differential privacy library
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 📧 Contact

For questions or collaborations:
- GitHub Issues: [Create an issue](https://github.com/Tanguyvans/cluster-shuffling-fl/issues)
- Email: [Your email]

---

**🚀 Ready to get started?** → [Quickstart Guide](docs/getting-started/quickstart.md)
