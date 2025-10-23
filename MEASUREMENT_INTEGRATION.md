# Gradient Pruning Measurement Integration

## Overview

Gradient pruning measurements are now **fully integrated** into the standard FL training workflow. When you run `main.py` with gradient pruning enabled, statistics are automatically tracked and saved to your results folder.

## What Gets Measured

When gradient pruning is enabled (`gradient_pruning.enabled = True` in `config.py`), the system automatically tracks:

1. **Compression Factor**: How many times smaller the pruned gradients are (e.g., 10x)
2. **Sparsity**: Percentage of gradients set to zero (e.g., 90%)
3. **Communication Savings**: Percentage of bandwidth saved (e.g., 89%)
4. **Parameters Transmitted**: Exact count of parameters sent vs. total

## Where Results Are Saved

All measurements are saved in your experiment's results folder:

```
results/[experiment_name]/
├── output.txt                      # Real-time training log with pruning stats
├── communication_metrics.txt       # Detailed communication analysis
├── config.json                     # Experiment configuration
└── ... (other standard outputs)
```

### 1. `output.txt` - Real-Time Training Log

Each client's training line now includes pruning information:

**Without Pruning:**
```
client c0_1: data:800 train: 800 train: 2.1234 0.45 val: 2.3456 0.42 test: 2.4567 0.40
```

**With Pruning:**
```
client c0_1: data:800 train: 800 train: 2.1234 0.45 val: 2.3456 0.42 test: 2.4567 0.40 | pruning: 9.2x compression, 89.1% saved
```

### 2. `communication_metrics.txt` - Detailed Analysis

This file contains comprehensive pruning statistics:

```
=== Communication Overhead Metrics ===

--- Protocol Communication ---
Total Protocol Communication: 0.00 MB
...

--- Storage Communication ---
Total Storage Communication: 0.00 MB
...

--- Gradient Pruning (DGC) ---
Status: ENABLED
Average Compression Factor: 9.2x
Average Sparsity: 89.1%
Average Communication Savings: 89.1%
Total Parameters Transmitted: 123,456 / 1,234,567
Number of Pruned Transmissions: 18

Estimated Communication Impact:
  Without Pruning: ~4.71 MB
  With Pruning: ~0.94 MB
  Saved: ~3.77 MB (80.0%)

=== Detailed Measurements ===

-- Gradient Pruning Statistics --
Round 1, c0_1: Compression 9.2x, Kept 12,345/101,010 params, Savings 89.1%
Round 1, c0_2: Compression 9.1x, Kept 12,456/101,010 params, Savings 88.9%
...
```

## How to Use

### Step 1: Enable Gradient Pruning

Edit `config.py`:

```python
"gradient_pruning": {
    "enabled": True,              # Enable pruning
    "keep_ratio": 0.1,            # 10% kept = 10x compression
    # ... other settings
}
```

### Step 2: Run Training

```bash
python3 main.py
```

### Step 3: Check Results

**During Training** - Monitor `output.txt`:
```bash
tail -f results/CFL/output.txt
```

You'll see pruning stats in real-time for each client.

**After Training** - Check `communication_metrics.txt`:
```bash
cat results/CFL/communication_metrics.txt
```

## Comparing With/Without Pruning

### Method 1: Run Two Experiments

**Experiment 1: Without Pruning**
```python
# config.py
"gradient_pruning": {"enabled": False}
```

```bash
python3 main.py
```

Results saved to: `results/CFL/`

**Experiment 2: With Pruning**
```python
# config.py
"gradient_pruning": {"enabled": True, "keep_ratio": 0.1}
```

```bash
python3 main.py
```

Results saved to: `results/CFL/` (or rename experiment)

**Compare:**
```bash
# Without pruning
cat results/CFL_baseline/communication_metrics.txt | grep "Status"

# With pruning
cat results/CFL_pruned/communication_metrics.txt | grep "Gradient Pruning" -A 10
```

### Method 2: Use Measurement Script

For quick comparisons, use the standalone measurement script:

```bash
# Simulate different keep ratios
python3 measure_communication.py --keep-ratio 0.1   # 10x compression
python3 measure_communication.py --keep-ratio 0.05  # 20x compression
python3 measure_communication.py --keep-ratio 0.01  # 100x compression
```

This generates comparison reports in `results/simulation_*/`

## Understanding the Metrics

### Compression Factor

- **What it is**: `Total parameters / Kept parameters`
- **Example**: `10.0x` means 10x fewer parameters sent
- **Good values**:
  - 5-10x: Balanced compression
  - 10-20x: Aggressive compression
  - 20-100x: Maximum compression (may affect accuracy)

### Sparsity

- **What it is**: Percentage of gradients set to zero
- **Example**: `90.0%` means 90% of gradients are pruned
- **Relationship**: `sparsity = 1 - keep_ratio`

### Communication Savings

- **What it is**: Percentage of bandwidth saved
- **Example**: `89.1%` means 89% less data transmitted
- **Note**: Slightly less than sparsity due to sparse format overhead (indices + values)

### Parameters Transmitted

- **What it is**: Exact count of non-zero parameters sent
- **Example**: `123,456 / 1,234,567` means 123K out of 1.2M params sent
- **Per client**: Each client sends this amount per round

## Example Output

Here's what you'll see with gradient pruning enabled:

### Terminal Output (During Training)

```
FlowerClient: Gradient Pruning ENABLED (keep_ratio=0.1)
...
[GradientPruning] Client c0_1: Compression 9.2x, Kept 12,345/101,010 params, Saved 89.1% communication
[GradientPruning] Client c0_2: Compression 9.1x, Kept 12,456/101,010 params, Saved 88.9% communication
...
```

### output.txt (Training Log)

```
client c0_1: data:800 train: 800 train: 2.1234 0.45 val: 2.3456 0.42 test: 2.4567 0.40 | pruning: 9.2x compression, 89.1% saved
client c0_2: data:800 train: 800 train: 2.0987 0.46 val: 2.2876 0.43 test: 2.3987 0.41 | pruning: 9.1x compression, 88.9% saved
```

### communication_metrics.txt (Summary)

```
--- Gradient Pruning (DGC) ---
Status: ENABLED
Average Compression Factor: 9.2x
Average Sparsity: 89.1%
Average Communication Savings: 89.1%
Total Parameters Transmitted: 123,456 / 1,234,567

Estimated Communication Impact:
  Without Pruning: ~4.71 MB
  With Pruning: ~0.94 MB
  Saved: ~3.77 MB (80.0%)
```

## For Research Papers

When reporting results, you can cite:

### Experiment Setup

> "Gradient pruning was applied with a keep ratio of 0.1, resulting in an average compression factor of 9.2x and 89.1% communication savings across all federated learning rounds."

### Results Format

| Configuration | Total Comm. | Compression | Savings |
|---------------|-------------|-------------|---------|
| Baseline | 4.71 MB | 1.0x | 0% |
| DGC (k=0.1) | 0.94 MB | 9.2x | 80.0% |
| DGC (k=0.05) | 0.50 MB | 18.5x | 89.4% |

### Metrics to Report

1. **Average Compression Factor** - From `communication_metrics.txt`
2. **Communication Savings (%)** - From `communication_metrics.txt`
3. **Impact on Accuracy** - Compare final test accuracy with/without pruning
4. **Convergence Speed** - Rounds to reach target accuracy

## Troubleshooting

### "No pruning statistics in output.txt"

**Cause**: Pruning is disabled in config

**Solution**:
```python
# config.py
"gradient_pruning": {"enabled": True}
```

### "Pruning section shows DISABLED"

**Cause**: No pruning measurements were recorded

**Solution**: Check that `gradient_pruning.enabled = True` before running

### "Communication savings seem low"

**Cause**: keep_ratio might be too high

**Solution**: Try lower keep_ratio values:
```python
"keep_ratio": 0.05  # Instead of 0.1
```

## Summary

✅ Gradient pruning measurements are **automatically integrated**
✅ Results appear in **standard output files** (`output.txt`, `communication_metrics.txt`)
✅ **No extra commands needed** - just run `python3 main.py`
✅ **Real-time monitoring** via `output.txt`
✅ **Comprehensive analysis** in `communication_metrics.txt`

That's it! Your gradient pruning impact is now fully measured and reported.
