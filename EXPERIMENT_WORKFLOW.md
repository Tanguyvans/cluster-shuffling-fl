# Experiment Workflow Guide

## How `run_paper_experiments.py` Works

The script automates **both training and attacks** for your paper experiments.

---

## 🔄 Complete Workflow (Automatic)

When you run:
```bash
python3 run_paper_experiments.py --experiment E1
```

**It automatically does:**

### **Phase 1: Training** ⚙️
1. Backs up your current `config.py`
2. Creates experiment-specific config (E1 settings)
3. Runs `python3 main.py` with FL training
4. Saves results to `results/E1_baseline/`
5. Saves gradients for rounds 1, 2, 3 (configured in script)

### **Phase 2: GIFD Attacks** 🎯 (Automatic)
1. For each client in `attack_clients` (default: `c0_1`, `c0_2`)
2. For each round in `attack_rounds` (default: rounds 1, 2, 3)
3. Automatically runs: `python3 attack_fl_ffhq.py --experiment E1_baseline --round X --client Y --attack-type gifd`
4. Saves attack results with PSNR metrics
5. Creates summary: `results/E1_baseline/attack_summary.json`

### **Phase 3: Cleanup** 🧹
1. Restores your original `config.py`
2. Prints summary with average PSNR

---

## 📊 Example Output

```bash
$ python3 run_paper_experiments.py --experiment E1

================================================================================
EXPERIMENT: E1
Description: Baseline - No defense mechanisms
================================================================================

================================================================================
TRAINING: E1 - Baseline - No defense mechanisms
================================================================================
  Clustering: False
  SMPC: none
  Gradient Pruning: False
  Rounds: 10
================================================================================

Starting FL training...
✓ Training completed successfully in 450.3s
  Results saved to: results/E1_baseline/

================================================================================
ATTACKS: E1 - Baseline - No defense mechanisms
================================================================================
  Clients: ['c0_1', 'c0_2']
  Rounds: [1, 2, 3]
================================================================================

  → Attacking c0_1 (Round 1)...
    ✓ Attack completed - PSNR: 28.5 dB

  → Attacking c0_1 (Round 2)...
    ✓ Attack completed - PSNR: 27.8 dB

  → Attacking c0_1 (Round 3)...
    ✓ Attack completed - PSNR: 29.1 dB

  → Attacking c0_2 (Round 1)...
    ✓ Attack completed - PSNR: 26.9 dB

  → Attacking c0_2 (Round 2)...
    ✓ Attack completed - PSNR: 28.3 dB

  → Attacking c0_2 (Round 3)...
    ✓ Attack completed - PSNR: 27.5 dB

✓ Attacks completed: 6/6 successful
  Average PSNR: 28.0 dB

✓ Attack summary saved to: results/E1_baseline/attack_summary.json

================================================================================
✓ EXPERIMENT E1 COMPLETED
================================================================================
```

---

## 🎯 Attack Details

### Where Attacks Happen (in the code)
Located in `run_paper_experiments.py` **lines 293-381**:

```python
def run_gifd_attack(experiment_id: str, client_id: str, round_num: int):
    """Run GIFD attack on a specific client/round"""

    # Line 308-317: Calls attack_fl_ffhq.py
    result = subprocess.run([
        "python3", "attack_fl_ffhq.py",
        "--experiment", experiment_name,
        "--round", str(round_num),
        "--client", client_id,
        "--attack-type", "gifd"
    ])
```

### Which Clients/Rounds are Attacked?
Configured in **lines 53-151** (`EXPERIMENT_CONFIGS`):

```python
"E1": {
    "attack_clients": ["c0_1", "c0_2"],  # Attack these clients
    "attack_rounds": [1, 2, 3],          # Attack these rounds
}
```

**You can customize this!** Change which clients/rounds to attack.

---

## 🛠️ Usage Options

### 1. Run Complete Experiment (Training + Attacks)
```bash
# Run single experiment (recommended for testing)
python3 run_paper_experiments.py --experiment E1

# Run all 8 experiments (for final paper results)
python3 run_paper_experiments.py --all
```

### 2. Run Training Only (Skip Attacks)
```bash
# Useful if you want to verify training works first
python3 run_paper_experiments.py --experiment E1 --training-only
```

### 3. Run Attacks Only (Skip Training)
```bash
# Useful if training already done, just re-run attacks
python3 run_paper_experiments.py --experiment E1 --attacks-only
```

### 4. List All Experiments
```bash
python3 run_paper_experiments.py --list
```

---

## 📂 Results Structure

After running `python3 run_paper_experiments.py --experiment E1`:

```
results/E1_baseline/
├── models/
│   ├── clients/
│   │   ├── round_001/
│   │   │   ├── c0_1_model.pt
│   │   │   ├── c0_1_gradients.pt        # ← Used for attacks
│   │   │   ├── c0_1_metadata.json
│   │   │   ├── c0_2_model.pt
│   │   │   ├── c0_2_gradients.pt        # ← Used for attacks
│   │   │   └── ...
│   │   ├── round_002/
│   │   └── round_003/
│   └── global/
├── logs/
├── config.json
├── output.txt
└── attack_summary.json                   # ← Attack results here!
```

**Attack-specific results** (in root directory):
```
fl_E1_baseline_r1_c0_1_gifd_results.png   # Visual comparison
fl_E1_baseline_r1_c0_1_gifd_metrics.json  # PSNR, SSIM, etc.
fl_E1_baseline_r1_c0_1_gifd_results.pth   # Full results
```

---

## 🎨 Customizing Attacks

### Change Which Clients to Attack

Edit `run_paper_experiments.py` (lines 63-64):

```python
"attack_clients": ["c0_1", "c0_2", "c0_3"],  # Add more clients
```

### Change Which Rounds to Attack

Edit `run_paper_experiments.py` (lines 64-65):

```python
"attack_rounds": [1, 2, 3, 5, 10],  # Attack different rounds
```

### Change Attack Type

Currently uses **GIFD** (line 316). You can change to:
- `"gifd"` - GIFD attack (GAN-based)
- `"gias"` - GIAS attack
- `"gradient_inversion"` - Standard gradient inversion

Edit line 316:
```python
"--attack-type", "gias"  # Change from "gifd"
```

---

## ⏱️ Time Estimates

Per experiment (on standard laptop):
- **Training**: ~5-10 minutes (10 rounds, CIFAR-10)
- **Attacks**: ~2-3 minutes per client/round (6 attacks = ~15 minutes)
- **Total per experiment**: ~20-25 minutes

**All 8 experiments**: ~2.5-3 hours

---

## 🚀 Quick Start for Your Paper

### Week 1: Core Experiments
```bash
# Day 1-2: Baseline and individual defenses
python3 run_paper_experiments.py --experiment E1  # Baseline
python3 run_paper_experiments.py --experiment E2  # Clustering
python3 run_paper_experiments.py --experiment E3  # SMPC Additive
python3 run_paper_experiments.py --experiment E4  # SMPC Shamir
python3 run_paper_experiments.py --experiment E5  # Pruning

# Day 3-4: Combined defenses
python3 run_paper_experiments.py --experiment E6  # Cluster + SMPC
python3 run_paper_experiments.py --experiment E7  # Cluster + SMPC
python3 run_paper_experiments.py --experiment E8  # Full defense
```

### Week 2: Analysis
```bash
# Generate comparison tables and plots
python3 analyze_results.py  # (I'll create this next!)
```

---

## 📊 Expected Results

| Experiment | PSNR (Expected) | Status |
|------------|----------------|---------|
| E1 (Baseline) | **>25 dB** | 🔴 VULNERABLE |
| E2 (Clustering) | ~20 dB | 🟡 WEAK |
| E3 (SMPC Additive) | ~18 dB | 🟢 BETTER |
| E4 (SMPC Shamir) | ~18 dB | 🟢 BETTER |
| E5 (Pruning) | ~22 dB | 🟡 SLIGHT |
| E6 (Cluster + SMPC) | **<15 dB** | ✅ STRONG |
| E7 (Cluster + SMPC) | **<15 dB** | ✅ STRONG |
| E8 (Full Defense) | **<15 dB** | ✅ STRONG |

**PSNR Interpretation:**
- **>25 dB**: High-quality reconstruction = **VULNERABLE**
- **20-25 dB**: Recognizable = **WEAK PROTECTION**
- **15-20 dB**: Moderate = **BETTER PROTECTION**
- **<15 dB**: Poor reconstruction = **STRONG PROTECTION**

---

## 🐛 Troubleshooting

### Problem: "Gradient file not found"
**Solution**: Training didn't save gradients. Check `save_gradients: True` in experiment config.

### Problem: "Attack failed"
**Possible causes:**
1. Missing `exploitai` library → Check `ls exploitai/`
2. GIFD imports fail → Try different attack type: `"gradient_inversion"`
3. CUDA/MPS issues → Force CPU in `attack_fl_ffhq.py` line 95

### Problem: Training takes too long
**Solution**: Reduce rounds in experiment config (line 62):
```python
"n_rounds": 5,  # Instead of 10
```

### Problem: Config not restored after crash
**Solution**: Manually restore:
```bash
cp config.py.backup config.py
```

---

## 📝 Summary

**The script does EVERYTHING automatically:**
1. ✅ Training with experiment-specific configs
2. ✅ Saving gradients for attack evaluation
3. ✅ Running GIFD attacks on all specified clients/rounds
4. ✅ Computing PSNR metrics
5. ✅ Saving results summary

**You just run:**
```bash
python3 run_paper_experiments.py --experiment E1
```

**And get:**
- Trained model
- Attack results with PSNR
- Ready for paper figures!
