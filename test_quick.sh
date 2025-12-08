#!/bin/bash
# Quick test of experiment pipeline
# This runs E1 with minimal parameters for fast testing

echo "================================================================================"
echo "QUICK TEST: E1 Baseline Experiment"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Rounds: 2 (instead of 10)"
echo "  - Epochs: 2 (instead of 5)"
echo "  - Attack client: c0_3"
echo "  - Attack rounds: [1, 2]"
echo "  - Attack types: GIFD and GIAS"
echo ""
echo "Expected time: ~5-10 minutes"
echo ""
echo "What will happen:"
echo "  1. Training FL model for 2 rounds (saves gradients)"
echo "  2. GIFD attack on c0_3 round 1"
echo "  3. GIAS attack on c0_3 round 2"
echo "  4. Save results to results/E1_baseline/"
echo ""
echo "================================================================================"
echo ""

read -p "Press Enter to start the test, or Ctrl+C to cancel..."

echo ""
echo "Starting experiment..."
echo ""

python3 run_paper_experiments.py --experiment E1 --training-only

echo ""
echo "================================================================================"
echo "RUNNING GRADIENT INVERSION ATTACKS"
echo "================================================================================"
echo ""

# Run gradient inversion attacks
echo "▶ Running Gradient Inversion attack on c0_3 round 1..."
python3 attack_fl_ffhq.py --experiment E1_baseline --round 1 --client c0_3 --attack-type gradient_inversion

echo ""
echo "▶ Running Gradient Inversion attack on c0_3 round 2..."
python3 attack_fl_ffhq.py --experiment E1_baseline --round 2 --client c0_3 --attack-type gradient_inversion

echo ""
echo "================================================================================"
echo "Test completed! Check results in:"
echo "  - results/E1_baseline/output.txt (training logs)"
echo "  - fl_E1_baseline_*.png (reconstructed images)"
echo "  - fl_E1_baseline_*_metrics.json (PSNR metrics)"
echo "================================================================================"
