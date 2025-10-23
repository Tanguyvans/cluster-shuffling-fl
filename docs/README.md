# Documentation

Welcome to the Cluster Shuffling Federated Learning documentation!

## 📚 Table of Contents

### 🚀 [Getting Started](getting-started/)
New to the project? Start here!

- **[Installation Guide](getting-started/installation.md)** - Setup and dependencies
- **[Quickstart Guide](getting-started/quickstart.md)** - Run your first FL experiment in 5 minutes
- **[Configuration Guide](getting-started/configuration.md)** - Complete config.py reference

### ✨ [Features](features/)
Core capabilities of the framework

- **[Gradient Pruning](features/gradient-pruning.md)** - Deep Gradient Compression (DGC) for 80% communication savings
- [Privacy Defenses](features/privacy-defenses.md) - SMPC, Differential Privacy, Cluster Shuffling
- [Aggregation Methods](features/aggregation-methods.md) - FedAvg, Krum, Trimmed Mean, FLTrust
- [Model Management](features/model-management.md) - Save, load, and organize models

### ⚔️ [Attacks](attacks/)
Poisoning and privacy attacks

**Poisoning Attacks** (Robustness):
- **[Poisoning Attacks Guide](attacks/poisoning-attacks.md)** - 6 attack types
  - Label Flipping, IPM, Sign Flip, Noise, ALIE, Backdoor

**Privacy Attacks** (Inference):
- **[Gradient Inversion Attacks](attacks/gradient-inversion.md)** - Reconstruct training data
  - DLG, iDLG, GIAS, **GIFD** (with StyleGAN2/BigGAN)
- [Membership Inference](attacks/membership-inference.md) - MIA attacks

### 📊 [Evaluation](evaluation/)
Systematic attack evaluation

- **[Attack Comparison](evaluation/attack-comparison.md)** - Compare pruned vs. full model attacks
- [Defense Evaluation](evaluation/defense-evaluation.md) - Test Krum, clustering, DP

### 📊 [Measurement](measurement/)
Track performance and overhead

- **[Communication Metrics](measurement/communication.md)** - Measure gradient pruning impact
- [Energy Tracking](measurement/energy-metrics.md) - Energy consumption monitoring
- [Performance Metrics](measurement/performance.md) - Time, throughput, convergence

### 🔧 [API Reference](api-reference/)
Developer documentation

- [ModelManager API](api-reference/model-manager.md) - Model saving and loading
- [MetricsTracker API](api-reference/metrics-tracker.md) - Metric collection
- [Client API](api-reference/client.md) - Client class reference
- [Server API](api-reference/server.md) - Server/Node class reference

### 🚢 [Deployment](deployment/)
Production deployment guides

- [Vast.AI Deployment](deployment/vastai.md) - Deploy on GPU cloud instances

### 📖 [Reference](reference/)
Additional resources

- [Commands](reference/commands.md) - Useful commands and scripts
- [File Structure](reference/file-structure.md) - Project organization
- [Datasets](reference/datasets.md) - CIFAR-10, FFHQ, Caltech256 details
- [Research Papers](reference/papers.md) - Citations and references

---

## Quick Links

### Most Common Tasks

🎯 **I want to...**

- **Run my first FL experiment** → [Quickstart Guide](getting-started/quickstart.md)
- **Reduce communication overhead** → [Gradient Pruning](features/gradient-pruning.md)
- **Test poisoning attacks** → [Poisoning Attacks](attacks/poisoning-attacks.md)
- **Evaluate gradient inversion** → [Gradient Inversion](attacks/gradient-inversion.md)
- **Enable privacy defenses** → [Privacy Defenses](features/privacy-defenses.md)
- **Understand config.py** → [Configuration Guide](getting-started/configuration.md)

### By Role

**👨‍🎓 Researchers**
- [Poisoning Attacks](attacks/poisoning-attacks.md) - Attack framework
- [Gradient Inversion](attacks/gradient-inversion.md) - Privacy attacks
- [Measurement](measurement/communication.md) - Metrics for papers

**👨‍💻 Developers**
- [API Reference](api-reference/) - Code documentation
- [Configuration Guide](getting-started/configuration.md) - All settings
- [File Structure](reference/file-structure.md) - Codebase organization

**🎓 Students**
- [Quickstart](getting-started/quickstart.md) - Quick tutorial
- [Gradient Pruning](features/gradient-pruning.md) - Communication efficiency
- [Privacy Defenses](features/privacy-defenses.md) - SMPC, DP, Clustering

---

## System Overview

### Architecture

```
┌─────────────┐
│   Clients   │ ─────┐
└─────────────┘      │
                     ├──► Cluster Shuffling
┌─────────────┐      │
│   Clients   │ ─────┤
└─────────────┘      │
                     ├──► SMPC Aggregation ──► Global Model
┌─────────────┐      │
│   Clients   │ ─────┤
└─────────────┘      │
                     ├──► Gradient Pruning (80% savings)
┌─────────────┐      │
│   Clients   │ ─────┘
└─────────────┘
```

### Key Features

✅ **Privacy Preservation**
- Cluster shuffling (dynamic reorganization)
- SMPC secret sharing (additive & Shamir)
- Differential privacy (DP-SGD)

✅ **Communication Efficiency**
- Gradient pruning (DGC) - 80% reduction
- Top-k sparsification with momentum
- Compatible with all privacy mechanisms

✅ **Attack Evaluation**
- 6 poisoning attack types
- Gradient inversion attacks
- Membership inference attacks
- Comprehensive metrics

✅ **Byzantine Robustness**
- Krum, Multi-Krum
- Trimmed Mean, Median
- FLTrust

---

## Documentation Status

| Section | Status | Description |
|---------|--------|-------------|
| Getting Started | ✅ Complete | Installation, quickstart, configuration |
| Gradient Pruning | ✅ Complete | DGC implementation guide |
| Poisoning Attacks | ✅ Complete | All 6 attack types documented |
| Gradient Inversion | ✅ Complete | Privacy attack evaluation |
| Pruned Models | ✅ Complete | Attack comparison framework |
| Communication Metrics | ✅ Complete | Measurement guide |
| Privacy Defenses | 🚧 Planned | SMPC, DP, Clustering deep dive |
| Aggregation Methods | 🚧 Planned | Krum, FLTrust, etc. |
| API Reference | 🚧 Planned | Code documentation |
| Deployment | ⚠️ Partial | Vast.AI guide available |

---

## Contributing

Found an issue or want to improve the docs?

1. Check existing documentation
2. Search [GitHub Issues](https://github.com/Tanguyvans/cluster-shuffling-fl/issues)
3. Create a pull request with improvements

---

## Need Help?

- **Getting Started**: [Quickstart Guide](getting-started/quickstart.md)
- **Configuration**: [Configuration Guide](getting-started/configuration.md)
- **Troubleshooting**: Check specific feature guides
- **GitHub**: [Issues](https://github.com/Tanguyvans/cluster-shuffling-fl/issues)

---

## License

This project is part of academic research. See [LICENSE](../LICENSE) for details.
