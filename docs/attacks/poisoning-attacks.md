# Poisoning Attacks

## Overview

The system includes a comprehensive poisoning attack framework to evaluate the robustness of federated learning defenses. Poisoning attacks attempt to degrade model performance or inject backdoors by manipulating training data or gradients from malicious clients.

## Attack Categories

### Data-Level Attacks
Manipulate training data before model training:
- **Label Flipping**: Corrupt training labels
- **Backdoor**: Insert triggers into training samples

### Gradient-Level Attacks
Manipulate gradients after local training:
- **IPM** (Inner Product Manipulation): Targeted gradient manipulation
- **Sign Flipping**: Flip gradient signs to disrupt aggregation
- **Noise Injection**: Add noise to gradients
- **ALIE**: "A Little Is Enough" byzantine attack

## Supported Attack Types

| Attack Type | Target | Impact | Stealthiness |
|-------------|--------|--------|--------------|
| **Label Flipping** | Data | High | Low |
| **IPM** | Gradients | High | Medium |
| **Sign Flipping** | Gradients | Medium | Low |
| **Noise Injection** | Gradients | Medium | Medium |
| **ALIE** | Gradients | High | High |
| **Backdoor** | Data | Critical | High |

---

## Configuration

### Basic Setup

Edit `config.py` to enable poisoning attacks:

```python
"poisoning_attacks": {
    "enabled": True,                            # Enable/disable poisoning attacks
    "malicious_clients": ["c0_1", "c0_2"],     # List of malicious client IDs
    "attack_type": "labelflip",                 # Attack type (see below)
    "attack_intensity": 0.2,                   # Attack strength (0.0 to 1.0)
    "attack_rounds": None,                     # Specific rounds to attack (None = all)
    "attack_frequency": 1.0,                   # Probability of attacking each round
}
```

---

## Attack Details

### 1. Label Flipping Attack

Flips training labels to degrade model accuracy or create targeted misclassification.

**Configuration:**

```python
"attack_type": "labelflip",
"labelflip_config": {
    "flip_type": "targeted",               # targeted, random, all_to_one
    "source_class": None,                  # Class to flip from (None = all)
    "target_class": 0,                     # Target class for flipping
    "num_classes": 10                      # Number of classes in dataset
}
```

**Flip Types:**
- **targeted**: Flip specific source class to target class (e.g., cat → dog)
- **random**: Randomly flip labels to any class
- **all_to_one**: Flip all labels to a single target class

**Example:**
```python
# Targeted attack: Flip all "airplane" labels to "bird"
"labelflip_config": {
    "flip_type": "targeted",
    "source_class": 0,     # Airplane
    "target_class": 2,     # Bird
    "num_classes": 10
}
```

**Defense Evaluation:**
- Byzantine-robust aggregation (Krum, Trimmed Mean)
- Cluster shuffling (dilutes attack across rounds)

---

### 2. IPM (Inner Product Manipulation)

Manipulates gradients to maximize interference with benign updates.

**Configuration:**

```python
"attack_type": "ipm",
"attack_intensity": 0.5,               # Manipulation strength (λ parameter)
"ipm_config": {
    "target_level": "cross_cluster",   # client, cross_cluster
    "lambda_param": 0.5,               # Manipulation strength
}
```

**Attack Levels:**
- **client**: Target gradients from other clients in same cluster
- **cross_cluster**: Target gradients from other clusters (more sophisticated)

**Algorithm:**
```
malicious_gradient = -λ * mean(benign_gradients)
```

**Defense Evaluation:**
- Cluster shuffling effectiveness
- Cross-cluster attack mitigation

---

### 3. Sign Flipping Attack

Flips gradient signs to disrupt model convergence.

**Configuration:**

```python
"attack_type": "signflip",
"signflip_config": {
    "flip_strategy": "random",         # random, all, selective
    "target_layers": None,             # Layers to target (None = all)
    "flip_probability": 0.3,           # Probability of flipping each gradient
    "magnitude_scaling": 1.0           # Scale factor for flipped gradients
}
```

**Flip Strategies:**
- **all**: Flip all gradient signs
- **random**: Randomly flip signs with probability
- **selective**: Flip only largest magnitude gradients

**Defense Evaluation:**
- Median/Trimmed Mean aggregation
- Sign-based attack detection

---

### 4. Noise Injection Attack

Adds noise to gradients to degrade model performance.

**Configuration:**

```python
"attack_type": "noise",
"noise_config": {
    "noise_type": "gaussian",          # gaussian, uniform, laplacian
    "noise_std": 0.1,                  # Standard deviation of noise
    "target_layers": None,             # Layers to target (None = all)
    "adaptive_noise": False            # Scale noise based on parameter magnitude
}
```

**Noise Types:**
- **gaussian**: Normal distribution N(0, σ²)
- **uniform**: Uniform distribution U(-a, a)
- **laplacian**: Laplacian distribution Lap(0, b)

**Adaptive Noise:**
```python
noise = noise_std * |gradient|  # Scale by gradient magnitude
```

**Defense Evaluation:**
- Gradient norm clipping
- Differential privacy (DP already adds noise)

---

### 5. ALIE (A Little Is Enough)

Sophisticated byzantine attack that stays close to benign gradient distribution.

**Configuration:**

```python
"attack_type": "alie",
"alie_config": {
    "deviation_type": "sign",          # sign, std, mean
    "aggregation_type": "mean",        # Expected aggregation method
    "epsilon": 0.1,                    # Small deviation parameter
    "num_malicious": 1                 # Number of malicious clients
}
```

**Deviation Types:**
- **sign**: Flip sign while maintaining similar magnitude
- **std**: Deviate by small standard deviation
- **mean**: Slight shift from mean gradient

**Algorithm:**
```
malicious_gradient = mean(benign) + ε * direction
```

**Defense Evaluation:**
- Most challenging attack to detect
- Tests robustness of all aggregation methods

---

### 6. Backdoor Attack

Inserts triggers into training data for targeted misclassification.

**Configuration:**

```python
"attack_type": "backdoor",
"backdoor_config": {
    "trigger_type": "pixel_pattern",       # pixel_pattern, square, cross, random_noise
    "trigger_size": 3,                     # Size of trigger pattern
    "trigger_position": "bottom_right",    # bottom_right, top_left, center, random
    "trigger_value": 1.0,                  # Trigger pixel intensity
    "backdoor_label": 0,                   # Target label for backdoor
    "poison_all_classes": True             # Poison samples from all classes
}
```

**Trigger Types:**
- **pixel_pattern**: Fixed pixel pattern (e.g., white square)
- **square**: Solid square trigger
- **cross**: Cross-shaped trigger
- **random_noise**: Random noise pattern

**Example:**
```python
# Backdoor: Images with white square in bottom-right → classify as "airplane"
"backdoor_config": {
    "trigger_type": "square",
    "trigger_size": 3,
    "trigger_position": "bottom_right",
    "trigger_value": 1.0,
    "backdoor_label": 0  # Airplane
}
```

**Defense Evaluation:**
- Cluster shuffling (backdoor dilution)
- Anomaly detection in gradients

---

## Running Attacks

### Step 1: Configure Attack

```python
# config.py
"poisoning_attacks": {
    "enabled": True,
    "malicious_clients": ["c0_1", "c0_2"],
    "attack_type": "ipm",
    "attack_intensity": 0.5,
}
```

### Step 2: Run Training

```bash
python3 main.py
```

### Step 3: Monitor Attack

The framework will:
1. Mark specified clients as malicious
2. Apply selected attack during training
3. Log attack effectiveness metrics
4. Evaluate defense mechanism performance

**Output Example:**
```
[Client c0_1] Applying IPMAttack attack in round 3
client c0_1: ... test: 1.234 45.67 (malicious - IPM attack)
```

---

## Attack Intensity Guidelines

| Intensity | Impact | Detectability | Use Case |
|-----------|--------|---------------|----------|
| 0.0 - 0.2 | Low | Hard to detect | Stealthy attacks |
| 0.2 - 0.5 | Medium | Moderate | Balanced evaluation |
| 0.5 - 0.8 | High | Easy to detect | Stress testing |
| 0.8 - 1.0 | Extreme | Obvious | Maximum impact testing |

---

## Defense Evaluation Workflow

### 1. Baseline (No Defense)

```python
"aggregation": {"method": "fedavg"},
"poisoning_attacks": {"enabled": True, "attack_type": "ipm"}
```

**Expected**: High attack success, degraded accuracy

### 2. Byzantine-Robust Aggregation

```python
"aggregation": {"method": "krum", "krum_malicious": 1},
"poisoning_attacks": {"enabled": True, "attack_type": "ipm"}
```

**Expected**: Reduced attack impact, better accuracy

### 3. Cluster Shuffling

```python
"clustering": True,
"poisoning_attacks": {"enabled": True, "attack_type": "ipm"}
```

**Expected**: Attack dilution across rounds

### 4. Combined Defense

```python
"aggregation": {"method": "krum"},
"clustering": True,
"diff_privacy": True,
"poisoning_attacks": {"enabled": True, "attack_type": "ipm"}
```

**Expected**: Maximum robustness

---

## Attack Effectiveness Metrics

### Model Performance
- **Training accuracy** on malicious clients
- **Global model accuracy** degradation
- **Per-class accuracy** (for targeted attacks)

### Attack Detection
- **Gradient norm** deviation from benign clients
- **Cosine similarity** between malicious and benign gradients
- **Aggregation rejection rate** (for robust aggregators)

### Defense Success
- **Accuracy recovery** with defense mechanisms
- **Attack mitigation rate**: (baseline_loss - defense_loss) / baseline_loss
- **Convergence speed** impact

---

## Research Use Cases

### Paper Experiment: "Defense Comparison"

```python
experiments = {
    "baseline": {
        "aggregation": "fedavg",
        "attack": "ipm",
        "intensity": 0.5
    },
    "krum": {
        "aggregation": "krum",
        "attack": "ipm",
        "intensity": 0.5
    },
    "clustering": {
        "clustering": True,
        "attack": "ipm",
        "intensity": 0.5
    },
    "combined": {
        "aggregation": "krum",
        "clustering": True,
        "diff_privacy": True,
        "attack": "ipm",
        "intensity": 0.5
    }
}
```

**Results Table:**
| Defense | Global Acc | Attack Impact | Convergence |
|---------|------------|---------------|-------------|
| None | 45% | -40% | Slow |
| Krum | 72% | -13% | Normal |
| Clustering | 68% | -17% | Normal |
| Combined | 81% | -4% | Normal |

---

## Implementation Details

### Attack Flow

```
1. Client marked as malicious (via config)
   ↓
2. Normal training on local data
   ↓
3. Attack applied:
   - Data-level: Poison training data before training
   - Gradient-level: Poison gradients after training
   ↓
4. Send poisoned updates to server
   ↓
5. Server applies defense (if enabled)
   ↓
6. Aggregate with benign clients
```

### Code Location

```
attacks/poisoning/
├── base_poisoning_attack.py      # Abstract base class
├── attack_factory.py             # Attack instantiation
├── labelflip_attack.py           # Label flipping
├── ipm_attack.py                 # IPM
├── signflip_attack.py            # Sign flipping
├── noise_attack.py               # Noise injection
├── alie_attack.py                # ALIE
├── backdoor_attack.py            # Backdoor
└── evaluation.py                 # Metrics & analysis
```

---

## Troubleshooting

### Issue: Attack has no effect

**Cause**: Intensity too low or robust aggregation filters it out

**Solution**: Increase attack_intensity or disable robust aggregation for baseline

### Issue: All clients affected

**Cause**: malicious_clients list incorrect

**Solution**: Check client IDs in config match actual client names (e.g., "c0_1", not "c1")

### Issue: Backdoor not working

**Cause**: Trigger not applied correctly

**Solution**: Verify trigger_size and trigger_position match dataset image size

---

## Summary

✅ **6 attack types** - Data and gradient poisoning
✅ **Configurable intensity** - From stealthy to obvious
✅ **Attack-specific configs** - Fine-grained control
✅ **Defense evaluation** - Test Krum, clustering, DP
✅ **Research-ready** - Systematic evaluation framework

For privacy attacks (gradient inversion, MIA), see:
- [Gradient Inversion Attacks](gradient-inversion.md)
- [Membership Inference Attacks](membership-inference.md)
