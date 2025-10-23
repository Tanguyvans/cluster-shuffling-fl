# Quickstart Guide

Get up and running with federated learning in 5 minutes!

## Prerequisites

- ✅ [Installation complete](installation.md)
- ✅ Python 3.8+
- ✅ Dependencies installed

## 1. Your First FL Training

### Run with Default Settings

```bash
python3 main.py
```

This will:
- Train on CIFAR-10 dataset
- Use 6 clients, 10 rounds
- Apply gradient pruning (80% communication savings)
- Save results to `results/cifar10_classic_c6_r10/`

**Expected Output:**
```
FlowerClient: Gradient Pruning ENABLED (keep_ratio=0.1)
...
[GradientPruning] Client c0_1: Compression 5.1x, Kept 6,095/62,006 params, Saved 80.3% communication
...
Round 10, Global aggregation: {'test_loss': 1.234, 'test_acc': 65.42}
```

Training takes ~10-15 minutes on CPU.

## 2. View Results

### Training Log

```bash
tail -f results/cifar10_classic_c6_r10/output.txt
```

**Example Output:**
```
client c0_1: train: 1.234 55.3% val: 1.456 52.1% test: 1.567 50.2% | pruning: 5.1x compression, 80.3% saved
client c0_2: train: 1.189 57.2% val: 1.398 54.3% test: 1.489 52.8% | pruning: 5.2x compression, 80.8% saved
```

### Communication Metrics

```bash
cat results/cifar10_classic_c6_r10/communication_metrics.txt
```

**Shows:**
- Gradient pruning statistics
- Communication savings
- Total bandwidth reduction

### Saved Models

```
results/cifar10_classic_c6_r10/models/
├── clients/round_010/        # Per-client models
├── pruned/round_010/         # Pruned models (for attack comparison)
└── global/round_010_global.pt # Final global model
```

## 3. Customize Your Experiment

### Change Dataset

```python
# config.py
"name_dataset": "cifar100"  # or "ffhq128", "caltech256"
```

### Adjust Training

```python
# config.py
"n_rounds": 20,           # More rounds
"batch_size": 64,         # Larger batches
"lr": 0.01,              # Higher learning rate
```

### Enable Privacy

```python
# config.py
"diff_privacy": True,     # Enable DP
"clustering": True,       # Enable cluster shuffling
```

## 4. Common Experiments

### Baseline (No Pruning)

```python
# config.py
"gradient_pruning": {"enabled": False}
```

```bash
python3 main.py
```

### With Privacy Defenses

```python
# config.py
"diff_privacy": True,
"clustering": True,
"gradient_pruning": {"enabled": True}
```

```bash
python3 main.py
```

### With Poisoning Attacks

```python
# config.py
"poisoning_attacks": {
    "enabled": True,
    "malicious_clients": ["c0_1"],
    "attack_type": "ipm",
    "attack_intensity": 0.5
}
```

```bash
python3 main.py
```

### Test Byzantine-Robust Aggregation

```python
# config.py
"aggregation": {
    "method": "krum",         # or "trimmed_mean", "median", "fltrust"
    "krum_malicious": 1
}
```

```bash
python3 main.py
```

## 5. Monitor Training

### Real-Time Monitoring

```bash
# In one terminal
python3 main.py

# In another terminal
watch -n 1 tail -20 results/cifar10_classic_c6_r10/output.txt
```

### Resource Usage

```bash
# Monitor CPU/GPU
htop

# Monitor GPU
nvidia-smi -l 1
```

## 6. Stop Training

Press `Ctrl+C` to gracefully stop training.

Models saved so far will be preserved in the results directory.

## 7. Quick Evaluation

### Test Attack Resistance

```bash
# Enable gradient saving
# config.py: "save_gradients": True, "save_gradients_rounds": [1,2,3]

# Run training
python3 main.py

# Run gradient inversion attack
python3 run_grad_inv.py --config default
```

### Compare Communication

```bash
# Simulate different compression ratios
python3 measure_communication.py --keep-ratio 0.1
python3 measure_communication.py --keep-ratio 0.05
```

## What's Next?

### Learn More
- [Configuration Guide](configuration.md) - Deep dive into config.py
- [Gradient Pruning](../features/gradient-pruning.md) - Communication efficiency
- [Poisoning Attacks](../attacks/poisoning-attacks.md) - Attack evaluation

### Advanced Topics
- [Privacy Defenses](../features/privacy-defenses.md) - SMPC, DP, Clustering
- [Gradient Inversion Attacks](../attacks/gradient-inversion.md) - Privacy attacks
- [Aggregation Methods](../features/aggregation-methods.md) - Krum, FLTrust, etc.

## Troubleshooting

### Training is slow

**Solution**: Reduce model size or use GPU
```python
"arch": "simplenet",  # Lighter model
"batch_size": 64      # Larger batches (if GPU available)
```

### Out of memory

**Solution**: Reduce batch size
```python
"batch_size": 16  # Smaller batches
```

### Accuracy is low

**Solution**: Train for more rounds or increase learning rate
```python
"n_rounds": 20,
"lr": 0.01
```

## Quick Reference

### Essential Commands

```bash
# Basic training
python3 main.py

# Test installation
python3 test_gradient_pruning.py

# Run attack evaluation
python3 run_grad_inv.py

# Measure communication
python3 measure_communication.py
```

### Essential Config Changes

```python
# Fast testing (1 round, small batches)
"n_rounds": 1,
"batch_size": 16,

# Production (many rounds, optimized)
"n_rounds": 50,
"batch_size": 64,
"lr": 0.01

# Privacy mode
"diff_privacy": True,
"clustering": True,

# Attack evaluation
"save_gradients": True,
"poisoning_attacks": {"enabled": True}
```

You're ready to start experimenting with privacy-preserving federated learning! 🚀
