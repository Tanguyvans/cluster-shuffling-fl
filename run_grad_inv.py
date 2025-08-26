#!/usr/bin/env python3
"""
Gradient Inversion Attack Runner - Single entry point for all attack scenarios

This script provides comprehensive control over gradient inversion attacks with
intelligent experiment discovery and flexible target selection.

Examples:
    # Attack latest experiment with default settings
    python run_grad_inv.py
    
    # Attack specific experiment
    python run_grad_inv.py --experiment cifar10_classic_c3_r3
    
    # Attack specific rounds and clients
    python run_grad_inv.py --rounds 1 2 --clients c0_1 c0_2
    
    # Quick test on latest experiment
    python run_grad_inv.py --config quick_test
    
    # Aggressive attack on specific targets
    python run_grad_inv.py --config aggressive --rounds 1 --clients c0_1
    
    # Compare attack configurations
    python run_grad_inv.py --compare default aggressive high_quality
"""

import argparse
import os
import sys
from pathlib import Path
import re
from attacks import GradientInversionAttacker, ATTACK_CONFIGS


def discover_experiments(results_dir="results"):
    """Discover all available experiments with metadata"""
    if not os.path.exists(results_dir):
        return []
    
    experiments = []
    for exp_name in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
            
        # Get experiment metadata
        exp_info = {
            'name': exp_name,
            'path': exp_path,
            'modified': os.path.getmtime(exp_path),
            'clients': set(),
            'rounds': set(),
            'has_gradients': False
        }
        
        # Scan for gradient files to get clients and rounds
        clients_dir = os.path.join(exp_path, "models", "clients")
        if os.path.exists(clients_dir):
            for round_dir in os.listdir(clients_dir):
                if not round_dir.startswith("round_"):
                    continue
                    
                # Extract round number
                round_match = re.match(r'round_(\d+)', round_dir)
                if round_match:
                    round_num = int(round_match.group(1))
                    exp_info['rounds'].add(round_num)
                
                # Scan for gradient files
                round_path = os.path.join(clients_dir, round_dir)
                if os.path.isdir(round_path):
                    for file in os.listdir(round_path):
                        if file.endswith('_gradients.pt'):
                            client_id = file.replace('_gradients.pt', '')
                            exp_info['clients'].add(client_id)
                            exp_info['has_gradients'] = True
        
        exp_info['rounds'] = sorted(list(exp_info['rounds']))
        exp_info['clients'] = sorted(list(exp_info['clients']))
        
        if exp_info['has_gradients']:  # Only include experiments with gradient data
            experiments.append(exp_info)
    
    # Sort by modification time (newest first)
    experiments.sort(key=lambda x: x['modified'], reverse=True)
    return experiments


def print_available_experiments(experiments):
    """Print available experiments in a nice format"""
    if not experiments:
        print("❌ No experiments with gradient data found in 'results/' directory")
        print("   Run training with 'save_gradients: True' in config.py first")
        return
    
    print(f"📁 Available Experiments ({len(experiments)} found):")
    print("=" * 80)
    print(f"{'#':<3} {'Name':<35} {'Clients':<15} {'Rounds':<15} {'Modified'}")
    print("-" * 80)
    
    for i, exp in enumerate(experiments):
        from datetime import datetime
        modified_str = datetime.fromtimestamp(exp['modified']).strftime('%Y-%m-%d %H:%M')
        clients_str = ', '.join(exp['clients'][:3]) + ('...' if len(exp['clients']) > 3 else '')
        rounds_str = ', '.join(map(str, exp['rounds']))
        
        marker = "⭐" if i == 0 else f"{i+1:2d}"
        print(f"{marker:<3} {exp['name']:<35} {clients_str:<15} {rounds_str:<15} {modified_str}")
    
    print("\n💡 Use --experiment <name> to select a specific experiment")
    print("   Or omit --experiment to use the latest (⭐)")


def select_experiment(exp_name, experiments):
    """Select experiment by name or index"""
    if not experiments:
        raise ValueError("No experiments available")
    
    if exp_name is None:
        return experiments[0]  # Return latest
    
    # Try by exact name first
    for exp in experiments:
        if exp['name'] == exp_name:
            return exp
    
    # Try by partial name match
    matches = [exp for exp in experiments if exp_name.lower() in exp['name'].lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"⚠️ Multiple experiments match '{exp_name}':")
        for exp in matches:
            print(f"   - {exp['name']}")
        raise ValueError("Please be more specific")
    
    # Try by index
    try:
        index = int(exp_name) - 1
        if 0 <= index < len(experiments):
            return experiments[index]
    except ValueError:
        pass
    
    raise ValueError(f"Experiment '{exp_name}' not found")


def validate_targets(experiment, rounds, clients):
    """Validate that requested rounds and clients exist"""
    available_rounds = set(experiment['rounds'])
    available_clients = set(experiment['clients'])
    
    errors = []
    
    if rounds:
        invalid_rounds = [r for r in rounds if r not in available_rounds]
        if invalid_rounds:
            errors.append(f"Invalid rounds: {invalid_rounds} (available: {sorted(available_rounds)})")
    
    if clients:
        invalid_clients = [c for c in clients if c not in available_clients]
        if invalid_clients:
            errors.append(f"Invalid clients: {invalid_clients} (available: {sorted(available_clients)})")
    
    if errors:
        raise ValueError("\n".join(errors))


def main():
    parser = argparse.ArgumentParser(
        description='Gradient Inversion Attack Runner - Complete control over attack scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Attack Configurations:
  quick_test    - Fast testing (1 restart, 8K iterations) 
  default       - Balanced attack (2 restarts, 24K iterations)
  aggressive    - Strong attack (5 restarts, 48K iterations) 
  conservative  - Light attack (1 restart, 12K iterations)
  high_quality  - Maximum quality (8 restarts, 60K iterations)

Target Selection:
  • No arguments = attack latest experiment, all clients, all rounds
  • --experiment <name> = select specific experiment
  • --rounds 1 2 3 = attack specific rounds only  
  • --clients c0_1 c0_2 = attack specific clients only
  • --clusters 0 1 = attack clients from specific clusters (if applicable)

Examples:
  %(prog)s                                    # Latest experiment, all targets
  %(prog)s --list                            # Show available experiments  
  %(prog)s --experiment cifar10_classic_c3_r3  # Specific experiment
  %(prog)s --config aggressive --rounds 1 2  # Strong attack on early rounds
  %(prog)s --compare default aggressive      # Compare configurations
        """
    )
    
    # Experiment selection
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments and exit')
    parser.add_argument('--experiment', '-e', type=str,
                       help='Experiment name, partial name, or number (default: latest)')
    
    # Target selection  
    parser.add_argument('--rounds', '-r', type=int, nargs='+',
                       help='Specific rounds to attack (default: all available)')
    parser.add_argument('--clients', '-c', type=str, nargs='+',
                       help='Specific clients to attack (default: all available)')
    parser.add_argument('--clusters', type=int, nargs='+',
                       help='Attack clients from specific clusters (extracts client IDs automatically)')
    
    # Attack configuration
    parser.add_argument('--config', choices=list(ATTACK_CONFIGS.keys()), default='default',
                       help='Attack configuration (default: %(default)s)')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Compare multiple attack configurations')
    
    # Output and execution
    parser.add_argument('--output', '-o', type=str, default='grad_inv_results',
                       help='Output directory (default: %(default)s)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='Computation device (default: %(default)s)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be attacked without running')
    
    args = parser.parse_args()
    
    # Welcome message
    print("🎯 Gradient Inversion Attack Runner")
    print("=" * 50)
    
    # Discover available experiments
    experiments = discover_experiments()
    
    # Handle list command
    if args.list:
        print_available_experiments(experiments)
        return 0
    
    if not experiments:
        print("❌ No experiments with gradient data found!")
        print("   1. Run training with 'save_gradients: True' in config.py")
        print("   2. Or check that 'results/' directory exists")
        return 1
    
    try:
        # Select target experiment
        selected_exp = select_experiment(args.experiment, experiments)
        print(f"🎯 Selected Experiment: {selected_exp['name']}")
        print(f"   📁 Path: {selected_exp['path']}")
        print(f"   👥 Clients: {len(selected_exp['clients'])} available")
        print(f"   🔄 Rounds: {len(selected_exp['rounds'])} available")
        
        # Determine target rounds and clients
        target_rounds = args.rounds if args.rounds else selected_exp['rounds']
        target_clients = args.clients if args.clients else selected_exp['clients']
        
        # Handle cluster-based client selection
        if args.clusters:
            cluster_clients = []
            for cluster_id in args.clusters:
                # Extract clients belonging to specific clusters (assumes naming pattern)
                cluster_pattern = f"c{cluster_id}_"
                cluster_clients.extend([c for c in selected_exp['clients'] if c.startswith(cluster_pattern)])
            
            if cluster_clients:
                target_clients = cluster_clients
                print(f"   🎪 Cluster filter: Found {len(cluster_clients)} clients in clusters {args.clusters}")
            else:
                print(f"   ⚠️ No clients found for clusters {args.clusters}")
        
        # Validate targets
        validate_targets(selected_exp, target_rounds, target_clients)
        
        print(f"\n📋 Attack Plan:")
        print(f"   🎯 Config: {args.config}")
        print(f"   🔄 Rounds: {target_rounds}")
        print(f"   👥 Clients: {target_clients}")
        print(f"   📊 Total attacks: {len(target_rounds) * len(target_clients)}")
        
        if args.dry_run:
            print(f"\n🔍 Dry run complete - no attacks performed")
            return 0
        
        # Initialize attacker
        attacker = GradientInversionAttacker(
            attack_config=args.config,
            device=args.device,
            output_dir=args.output
        )
        
        # Handle comparison mode
        if args.compare:
            print(f"\n🔬 Comparison Mode: {args.compare}")
            results = attacker.compare_configurations(
                args.compare, 
                selected_exp['path']
            )
            return 0
        
        # Execute attacks
        print(f"\n🚀 Starting gradient inversion attacks...")
        results = attacker.attack_experiment(
            exp_dir=selected_exp['path'],
            attack_rounds=target_rounds,
            clients=target_clients
        )
        
        if results:
            success_count = len([r for r in results if r['psnr'] > 20])
            avg_psnr = sum(r['psnr'] for r in results) / len(results)
            
            print(f"\n✅ Attack Complete!")
            print(f"   📊 Results: {len(results)} attacks performed")
            print(f"   📈 Average PSNR: {avg_psnr:.2f} dB")
            print(f"   🎯 Successful attacks (>20 dB): {success_count}/{len(results)}")
            print(f"   📁 Results saved to: {args.output}/")
        else:
            print(f"\n❌ No successful attacks performed")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())