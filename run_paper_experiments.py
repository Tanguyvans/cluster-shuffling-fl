#!/usr/bin/env python3
"""
Comprehensive Experiment Script for Paper
==========================================

Trains FL models with different configurations (baseline, MPC, pruning, combined)
then performs GIFD gradient inversion attacks on selected clients.

Usage:
    # Run all experiments
    python3 run_paper_experiments.py --all

    # Run specific experiment
    python3 run_paper_experiments.py --experiment E1

    # Run training only (no attacks)
    python3 run_paper_experiments.py --experiment E1 --training-only

    # Run attacks only (assumes training done)
    python3 run_paper_experiments.py --experiment E1 --attacks-only

Experiments:
    E1: Baseline (no defense)
    E2: Clustering only
    E3: SMPC (Additive) only
    E4: SMPC (Shamir) only
    E5: Gradient pruning only
    E6: Clustering + SMPC (Additive)
    E7: Clustering + SMPC (Shamir)
    E8: Full defense (Clustering + SMPC + Pruning)
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
import copy

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import settings as base_settings

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENT_CONFIGS = {
    "E1": {
        "name": "E1_baseline",
        "description": "Baseline - No defense mechanisms",
        "clustering": False,
        "type_ss": "none",
        "gradient_pruning_enabled": False,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2],     # Just 2 rounds for testing
        "n_rounds": 2,                       # Quick test: only 2 rounds
        "n_epochs": 2,                       # Quick test: only 2 epochs
        "attack_clients": ["c0_3"],          # Attack client c0_3 (has gradients saved)
        "attack_rounds": [1, 2],             # Attack rounds 1, 2
    },
    "E2": {
        "name": "E2_clustering_only",
        "description": "Clustering only",
        "clustering": True,
        "type_ss": "none",
        "gradient_pruning_enabled": False,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2, 3],
        "n_rounds": 10,
        "attack_clients": ["c0_1", "c0_2"],
        "attack_rounds": [1, 2, 3],
    },
    "E3": {
        "name": "E3_smpc_additive",
        "description": "SMPC (Additive) only",
        "clustering": False,
        "type_ss": "additif",
        "gradient_pruning_enabled": False,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2, 3],
        "n_rounds": 10,
        "attack_clients": ["c0_1", "c0_2"],
        "attack_rounds": [1, 2, 3],
    },
    "E4": {
        "name": "E4_smpc_shamir",
        "description": "SMPC (Shamir) only",
        "clustering": False,
        "type_ss": "shamir",
        "gradient_pruning_enabled": False,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2, 3],
        "n_rounds": 10,
        "attack_clients": ["c0_1", "c0_2"],
        "attack_rounds": [1, 2, 3],
    },
    "E5": {
        "name": "E5_pruning_only",
        "description": "Gradient pruning (10% keep) only",
        "clustering": False,
        "type_ss": "none",
        "gradient_pruning_enabled": True,
        "gradient_pruning_keep_ratio": 0.1,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2, 3],
        "n_rounds": 10,
        "attack_clients": ["c0_1", "c0_2"],
        "attack_rounds": [1, 2, 3],
    },
    "E6": {
        "name": "E6_cluster_smpc_additive",
        "description": "Clustering + SMPC (Additive)",
        "clustering": True,
        "type_ss": "additif",
        "gradient_pruning_enabled": False,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2, 3],
        "n_rounds": 10,
        "attack_clients": ["c0_1", "c0_2"],
        "attack_rounds": [1, 2, 3],
    },
    "E7": {
        "name": "E7_cluster_smpc_shamir",
        "description": "Clustering + SMPC (Shamir)",
        "clustering": True,
        "type_ss": "shamir",
        "gradient_pruning_enabled": False,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2, 3],
        "n_rounds": 10,
        "attack_clients": ["c0_1", "c0_2"],
        "attack_rounds": [1, 2, 3],
    },
    "E8": {
        "name": "E8_full_defense",
        "description": "Full defense: Clustering + SMPC (Shamir) + Pruning",
        "clustering": True,
        "type_ss": "shamir",
        "gradient_pruning_enabled": True,
        "gradient_pruning_keep_ratio": 0.1,
        "save_gradients": True,
        "save_gradients_rounds": [1, 2, 3],
        "n_rounds": 10,
        "attack_clients": ["c0_1", "c0_2"],
        "attack_rounds": [1, 2, 3],
    },
}


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def create_experiment_config(experiment_id: str) -> Dict:
    """Create config.py content for an experiment"""

    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_id}")

    exp_config = EXPERIMENT_CONFIGS[experiment_id]
    config = copy.deepcopy(base_settings)

    # Override with experiment-specific settings
    config["clustering"] = exp_config["clustering"]
    config["type_ss"] = exp_config["type_ss"]
    config["save_gradients"] = exp_config["save_gradients"]
    config["save_gradients_rounds"] = exp_config["save_gradients_rounds"]
    config["n_rounds"] = exp_config["n_rounds"]

    # Gradient pruning
    config["gradient_pruning"]["enabled"] = exp_config["gradient_pruning_enabled"]
    if "gradient_pruning_keep_ratio" in exp_config:
        config["gradient_pruning"]["keep_ratio"] = exp_config["gradient_pruning_keep_ratio"]

    # Set experiment name for results directory
    config["save_results"] = f"results/{exp_config['name']}/"

    # Ensure consistent training settings
    config["n_epochs"] = exp_config.get("n_epochs", 5)  # Use experiment-specific or default to 5
    config["batch_size"] = 32
    config["diff_privacy"] = False  # No DP for paper
    config["poisoning_attacks"]["enabled"] = False  # No poisoning during training

    return config


def backup_config():
    """Backup current config.py"""
    if os.path.exists("config.py"):
        shutil.copy("config.py", "config.py.backup")
        print("✓ Backed up config.py to config.py.backup")


def restore_config():
    """Restore original config.py"""
    if os.path.exists("config.py.backup"):
        shutil.copy("config.py.backup", "config.py")
        print("✓ Restored original config.py")


def write_config_file(config: Dict):
    """Write config dictionary to config.py"""

    config_content = f"""import os
from pathlib import Path

# Auto-generated config for experiment
# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

DATASET_ROOT = os.path.join(os.path.dirname(__file__), 'dataset')

DATASET_PATHS = {{
    'cifar10': os.path.join(DATASET_ROOT, 'cifar10'),
    'cifar100': os.path.join(DATASET_ROOT, 'cifar-100-python'),
    'ffhq128': os.path.join(DATASET_ROOT, 'ffhq_dataset'),
    'caltech256': os.path.join(DATASET_ROOT, 'caltech256'),
}}

settings = {repr(config)}
"""

    with open("config.py", "w") as f:
        f.write(config_content)

    print(f"✓ Written config.py for experiment")

    # Force Python to see the new config by removing any cached .pyc files
    import subprocess
    subprocess.run(["find", ".", "-name", "*.pyc", "-delete"], capture_output=True)
    if os.path.exists("__pycache__"):
        subprocess.run(["rm", "-rf", "__pycache__"], capture_output=True)


# ============================================================================
# TRAINING
# ============================================================================

def run_training(experiment_id: str) -> bool:
    """Run FL training for an experiment"""

    exp_config = EXPERIMENT_CONFIGS[experiment_id]

    print("\n" + "=" * 80)
    print(f"TRAINING: {experiment_id} - {exp_config['description']}")
    print("=" * 80)
    print(f"  Clustering: {exp_config['clustering']}")
    print(f"  SMPC: {exp_config['type_ss']}")
    print(f"  Gradient Pruning: {exp_config['gradient_pruning_enabled']}")
    print(f"  Rounds: {exp_config['n_rounds']}")
    print("=" * 80)

    # Build command-line arguments for main.py
    cmd = [
        "python3", "main.py",
        "--experiment", exp_config['name'],
        "--rounds", str(exp_config['n_rounds']),
        "--epochs", str(exp_config['n_epochs']),
        "--clustering", "true" if exp_config['clustering'] else "false",
        "--smpc", exp_config['type_ss'],
        "--pruning", "true" if exp_config['gradient_pruning_enabled'] else "false",
        "--save-gradients", "true" if exp_config['save_gradients'] else "false",
        "--save-gradients-rounds", ",".join(map(str, exp_config['save_gradients_rounds']))
    ]

    # Run main.py with arguments (no config.py modification needed!)
    print("\nStarting FL training with command:")
    print(f"  {' '.join(cmd)}")
    print("\n(Training output will be shown below...)\n")
    start_time = time.time()

    try:
        # Pass arguments directly to main.py
        result = subprocess.run(
            cmd,
            timeout=3600  # 1 hour timeout
        )

        training_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n{'='*80}")
            print(f"✓ Training completed successfully in {training_time:.1f}s")
            print(f"  Results saved to: results/{exp_config['name']}/")
            print(f"{'='*80}\n")
            return True
        else:
            print(f"\n{'='*80}")
            print(f"✗ Training failed with return code {result.returncode}")
            print(f"{'='*80}\n")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n✗ Training timed out after 1 hour")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted - Checking if training completed...")
        # Check if training completed successfully despite interrupt
        results_exist = os.path.exists(f"results/{exp_config['name']}/output.txt")
        if results_exist:
            print(f"✓ Training output found - continuing to attacks\n")
            return True
        print(f"✗ Training incomplete")
        return False
    except Exception as e:
        print(f"\n✗ Training failed with exception: {e}")
        return False


# ============================================================================
# GIFD ATTACKS
# ============================================================================

def run_gifd_attack(experiment_id: str, client_id: str, round_num: int) -> Optional[Dict]:
    """Run GIFD attack on a specific client/round"""

    exp_config = EXPERIMENT_CONFIGS[experiment_id]
    experiment_name = exp_config['name']

    print(f"\n  → Attacking {client_id} (Round {round_num})...")

    # Check if gradient file exists
    gradient_file = Path(f"results/{experiment_name}/models/clients/round_{round_num:03d}/{client_id}_gradients.pt")

    if not gradient_file.exists():
        print(f"    ✗ Gradient file not found: {gradient_file}")
        return None

    # Run GIFD attack using attack_fl_ffhq.py (works for CIFAR too)
    try:
        # Show attack output in real-time
        result = subprocess.run(
            [
                "python3", "attack_fl_ffhq.py",
                "--experiment", experiment_name,
                "--round", str(round_num),
                "--client", client_id,
                "--attack-type", "gifd"
            ],
            timeout=600  # 10 minute timeout per attack
        )

        if result.returncode == 0:
            # Load attack metrics
            metrics_file = Path(f"fl_{experiment_name}_r{round_num}_{client_id}_gifd_metrics.json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                psnr = metrics.get('psnr', 0)
                print(f"\n    ✓ Attack completed - PSNR: {psnr:.2f} dB\n")
                return metrics
            else:
                print(f"\n    ✓ Attack completed (no metrics file)\n")
                return {"success": True}
        else:
            print(f"\n    ✗ Attack failed\n")
            return None

    except subprocess.TimeoutExpired:
        print(f"    ✗ Attack timed out")
        return None
    except Exception as e:
        print(f"    ✗ Attack failed with exception: {e}")
        return None


def run_all_attacks(experiment_id: str) -> Dict[str, Dict]:
    """Run GIFD attacks on all specified clients and rounds"""

    exp_config = EXPERIMENT_CONFIGS[experiment_id]

    print("\n" + "=" * 80)
    print(f"ATTACKS: {experiment_id} - {exp_config['description']}")
    print("=" * 80)
    print(f"  Clients: {exp_config['attack_clients']}")
    print(f"  Rounds: {exp_config['attack_rounds']}")
    print("=" * 80)

    attack_results = {}

    for client_id in exp_config['attack_clients']:
        for round_num in exp_config['attack_rounds']:
            key = f"{client_id}_r{round_num}"
            metrics = run_gifd_attack(experiment_id, client_id, round_num)
            attack_results[key] = metrics

    # Summary
    successful_attacks = sum(1 for m in attack_results.values() if m is not None)
    total_attacks = len(attack_results)

    print(f"\n✓ Attacks completed: {successful_attacks}/{total_attacks} successful")

    # Calculate average PSNR
    psnr_values = [m.get('psnr', 0) for m in attack_results.values() if m is not None and 'psnr' in m]
    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        print(f"  Average PSNR: {avg_psnr:.2f} dB")

    return attack_results


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_experiment(experiment_id: str, training_only: bool = False, attacks_only: bool = False) -> bool:
    """Run complete experiment: training + attacks"""

    if experiment_id not in EXPERIMENT_CONFIGS:
        print(f"✗ Unknown experiment: {experiment_id}")
        print(f"  Available: {', '.join(EXPERIMENT_CONFIGS.keys())}")
        return False

    exp_config = EXPERIMENT_CONFIGS[experiment_id]

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_id}")
    print(f"Description: {exp_config['description']}")
    print("=" * 80)

    success = True

    # Training phase
    if not attacks_only:
        training_success = run_training(experiment_id)
        if not training_success:
            print(f"\n✗ Experiment {experiment_id} failed at training stage")
            return False

    # Attack phase
    if not training_only:
        attack_results = run_all_attacks(experiment_id)

        # Save attack results summary
        results_dir = Path(f"results/{exp_config['name']}")
        results_dir.mkdir(parents=True, exist_ok=True)

        summary_file = results_dir / "attack_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(attack_results, f, indent=2)

        print(f"\n✓ Attack summary saved to: {summary_file}")

    print(f"\n{'=' * 80}")
    print(f"✓ EXPERIMENT {experiment_id} COMPLETED")
    print(f"{'=' * 80}\n")

    return True


def run_all_experiments():
    """Run all experiments sequentially"""

    print("\n" + "=" * 80)
    print("RUNNING ALL EXPERIMENTS FOR PAPER")
    print("=" * 80)
    print(f"Total experiments: {len(EXPERIMENT_CONFIGS)}")
    print("=" * 80)

    results = {}
    start_time = time.time()

    for exp_id in sorted(EXPERIMENT_CONFIGS.keys()):
        exp_start = time.time()
        success = run_experiment(exp_id)
        exp_time = time.time() - exp_start

        results[exp_id] = {
            "success": success,
            "time": exp_time
        }

        print(f"\n{exp_id}: {'✓ SUCCESS' if success else '✗ FAILED'} ({exp_time/60:.1f} minutes)")

    total_time = time.time() - start_time

    # Final summary
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS SUMMARY")
    print("=" * 80)

    for exp_id, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {exp_id}: {result['time']/60:.1f} minutes")

    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nTotal: {successful}/{len(results)} successful")
    print(f"Total time: {total_time/3600:.1f} hours")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive experiments for paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--experiment', type=str,
                       help='Run specific experiment (E1, E2, ..., E8)')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--training-only', action='store_true',
                       help='Run training only (skip attacks)')
    parser.add_argument('--attacks-only', action='store_true',
                       help='Run attacks only (skip training)')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\nAvailable Experiments:")
        print("=" * 80)
        for exp_id, exp_config in sorted(EXPERIMENT_CONFIGS.items()):
            print(f"\n{exp_id}: {exp_config['description']}")
            print(f"  Clustering: {exp_config['clustering']}")
            print(f"  SMPC: {exp_config['type_ss']}")
            print(f"  Gradient Pruning: {exp_config['gradient_pruning_enabled']}")
        print("\n" + "=" * 80)
        return

    # Backup original config
    backup_config()

    try:
        if args.all:
            run_all_experiments()
        elif args.experiment:
            run_experiment(
                args.experiment,
                training_only=args.training_only,
                attacks_only=args.attacks_only
            )
        else:
            parser.print_help()

    finally:
        # Restore original config
        restore_config()


if __name__ == "__main__":
    main()
