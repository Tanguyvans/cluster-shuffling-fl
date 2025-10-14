# 🎯 Gradient Inversion Attack Guide

This guide shows how to use the modular gradient inversion attack framework to evaluate privacy protection in your federated learning experiments.

## Quick Start

### 1. List Available Experiments
```bash
python run_grad_inv.py --list
```
Shows all experiments with gradient data and their metadata.

### 2. Attack Latest Experiment (Default)
```bash
python run_grad_inv.py
```
Attacks the most recent experiment with all clients and rounds.

### 3. Quick Test
```bash
python run_grad_inv.py --config quick_test
```
Fast attack for testing (1 restart, 8K iterations).

## Target Selection

### Select Specific Experiment
```bash
# By exact name
python run_grad_inv.py --experiment cifar10_classic_c3_r3

# By partial name (if unique)
python run_grad_inv.py --experiment cifar10_classic
```

### Select Specific Rounds and Clients
```bash
# Attack only rounds 1 and 2
python run_grad_inv.py --rounds 1 2

# Attack specific clients
python run_grad_inv.py --clients c0_1 c0_2

# Attack early rounds on specific clients
python run_grad_inv.py --rounds 1 2 --clients c0_1 c0_2
```

### Cluster-Based Selection
```bash
# Attack all clients from cluster 0
python run_grad_inv.py --clusters 0

# Attack clients from clusters 0 and 1  
python run_grad_inv.py --clusters 0 1
```

## Attack Configurations

| Config | Restarts | Iterations | Use Case |
|--------|----------|------------|----------|
| `quick_test` | 1 | 8,000 | Fast testing |
| `default` | 2 | 24,000 | Balanced evaluation |
| `aggressive` | 5 | 48,000 | Strong attack |
| `conservative` | 1 | 12,000 | Light attack |
| `high_quality` | 8 | 60,000 | Maximum quality |

### Examples
```bash
# Strong attack for maximum reconstruction quality
python run_grad_inv.py --config aggressive

# Conservative attack to test basic vulnerability
python run_grad_inv.py --config conservative --rounds 1
```

## Configuration Comparison

Compare multiple attack configurations on the same experiment:

```bash
# Compare default vs aggressive
python run_grad_inv.py --compare default aggressive

# Compare all configurations
python run_grad_inv.py --compare quick_test default aggressive conservative high_quality
```

## Advanced Examples

### Research Scenarios

```bash
# Test vulnerability of early training rounds
python run_grad_inv.py --config aggressive --rounds 1 2 3

# Focus attack on specific client
python run_grad_inv.py --config high_quality --clients c0_1

# Quick vulnerability check across all experiments
python run_grad_inv.py --config quick_test
```

### Systematic Privacy Evaluation

```bash
# 1. Train baseline (no privacy)
python main.py  # with privacy mechanisms disabled

# 2. Attack baseline
python run_grad_inv.py --config aggressive --output baseline_attack

# 3. Train with SMPC enabled  
# (modify config.py: clustering=True)
python main.py

# 4. Attack SMPC experiment
python run_grad_inv.py --config aggressive --output smpc_attack

# 5. Compare results
ls baseline_attack/ smpc_attack/
```

## Dry Run Mode

Test your attack plan without executing:
```bash
python run_grad_inv.py --dry-run --rounds 1 2 --clients c0_1
```

Shows exactly what would be attacked.

## Output and Results

### Result Structure
```
grad_inv_results/
├── round_1_client_c0_1_comparison.png    # Side-by-side comparison
├── round_1_client_c0_1_original.png      # Original images
├── round_1_client_c0_1_reconstructed.png # Reconstructed images
├── individual/                           # Individual image files
└── attack_results.pt                     # Detailed metrics
```

### Understanding Results

- **PSNR > 25 dB**: High-quality reconstruction (vulnerable)
- **PSNR 20-25 dB**: Good reconstruction (weak protection) 
- **PSNR 15-20 dB**: Moderate reconstruction (fair protection)
- **PSNR < 15 dB**: Poor reconstruction (strong protection)

## Integration with Privacy Mechanisms

### Test Different Privacy Settings

1. **No Privacy** (Baseline):
   ```python
   # config.py
   "clustering": False,
   "diff_privacy": False
   ```

2. **SMPC Only**:
   ```python
   # config.py  
   "clustering": True,
   "diff_privacy": False
   ```

3. **Differential Privacy Only**:
   ```python
   # config.py
   "clustering": False, 
   "diff_privacy": True
   ```

4. **Combined Protection**:
   ```python
   # config.py
   "clustering": True,
   "diff_privacy": True
   ```

### Systematic Evaluation Workflow

```bash
# For each privacy setting:
# 1. Train with that configuration
python main.py

# 2. Attack with multiple intensities  
python run_grad_inv.py --compare quick_test default aggressive

# 3. Analyze protection effectiveness
# (Check average PSNR and success rates in output)
```

This gives you comprehensive privacy evaluation across different attack intensities and defense mechanisms.

---

## 🖼️ FFHQ (Face) Dataset Attacks

For gradient inversion attacks on **FFHQ (Flickr-Faces-High-Quality) datasets**, use the dedicated `attack_fl_ffhq.py` script. This script supports specialized attack types optimized for face reconstruction.

### Attack Types Available

1. **GIAS** - Gradient Inversion Attack with Adaptive Selection
2. **GIFD** - Gradient Inversion from Federated Data with generative priors
3. **Gradient Inversion** - Standard reconstruction attack

### Quick Start Examples

```bash
# Attack with default settings (auto-detects experiment)
python3 attack_fl_ffhq.py --attack-type gias

# Attack specific experiment, round, and client
python3 attack_fl_ffhq.py --experiment ffhq128_classic_c6_r3 --round 1 --client c0_1 --attack-type gifd

# Attack using direct model path
python3 attack_fl_ffhq.py --model-path results/ffhq128_classic_c6_r3 --round 2 --client c0_3 --attack-type gradient_inversion
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment` | str | auto-detect | Experiment name (directory in results/) |
| `--round` | int | 1 | FL round to attack |
| `--client` | str | c0_1 | Client ID to attack |
| `--attack-type` | str | gias | Attack type: gias, gifd, gradient_inversion |
| `--model-path` | str | None | Direct path to experiment directory |

### FFHQ Attack Examples

#### GIAS Attack (High-Quality Reconstruction)

```bash
# GIAS attack with face-optimized configuration
python3 attack_fl_ffhq.py --experiment ffhq128_classic_c6_r3 --attack-type gias --round 2 --client c0_1

# Attack early rounds to test privacy leakage in first training stages
python3 attack_fl_ffhq.py --attack-type gias --rounds 1 2 3
```

#### GIFD Attack (Generative Model Prior)

```bash
# GIFD attack using StyleGAN2/BigGAN for enhanced face reconstruction
python3 attack_fl_ffhq.py --experiment ffhq128_classic_c6_r3 --attack-type gifd --round 2 --client c0_1
```

#### Standard Gradient Inversion

```bash
# Traditional gradient inversion on FFHQ models
python3 attack_fl_ffhq.py --attack-type gradient_inversion --round 1 --client c0_2
```

### Auto-Detection Features

The script automatically:

- Detects the latest FFHQ experiment if none specified
- Identifies correct model input dimensions (128x128 for FFHQ)
- Uses FFHQ-specific normalization parameters
- Loads compatible model architectures

### Output Files

FFHQ attacks generate experiment-specific output files:

```text
fl_{experiment}_r{round}_{client}_{attack_type}_results.png  # Visual comparison
fl_{experiment}_r{round}_{client}_{attack_type}_results.pth  # Full results data
fl_{experiment}_r{round}_{client}_{attack_type}_metrics.json # Numerical metrics
```

### Example Workflow

```bash
# 1. Train FFHQ models with privacy defenses
python3 main.py  # with name_dataset: "ffhq128" in config.py

# 2. Attack with multiple methods
python3 attack_fl_ffhq.py --attack-type gias --round 1
python3 attack_fl_ffhq.py --attack-type gifd --round 1
python3 attack_fl_ffhq.py --attack-type gradient_inversion --round 1

# 3. Compare attack quality metrics
cat fl_*_metrics.json | grep psnr
```

🎯 **FFHQ vs CIFAR Attacks**: Use `attack_fl_ffhq.py` for face datasets and `run_grad_inv.py` for CIFAR/non-face datasets. The FFHQ script is optimized for face reconstruction quality (128x128).