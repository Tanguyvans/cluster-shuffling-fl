"""
Smart Federated Attack - Prioritizes gradients with highest vulnerability
Targets rounds/clients with highest loss and gradient norms for best reconstruction
Adapted for cluster-shuffling-fl framework
"""

import torch
import torchvision
import numpy as np
import os
import time
import sys
import inversefed
from collections import defaultdict

# Import from existing framework
from models.factory import Net
import config as cfg

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = cfg.settings['arch']  # Use framework's model architecture
NUM_IMAGES = 10
NUM_CLIENTS = cfg.settings['number_of_clients_per_node']

# Enhanced attack config for vulnerable gradients
SMART_ATTACK_CONFIG = dict(
    signed=True,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal',
    lr=0.15,  # Higher LR for high-loss gradients
    optim='adam',
    restarts=3,  # More restarts for better results
    max_iterations=30000,  # More iterations for high-loss cases
    total_variation=1e-2,
    init='randn',
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

# Standard config for low-vulnerability gradients
STANDARD_ATTACK_CONFIG = dict(
    signed=True,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal',
    lr=0.1,
    optim='adam',
    restarts=2,
    max_iterations=24000,
    total_variation=1e-2,
    init='randn',
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

def analyze_gradient_vulnerability():
    """Analyze all available gradients to find most vulnerable targets"""
    vulnerability_scores = []
    available_files = []
    
    # Look for gradient files in the framework's directory structure
    gradient_dir = f'results/gradient_inversion/{cfg.settings["arch"]}_{cfg.settings["name_dataset"]}'
    
    if not os.path.exists(gradient_dir):
        print(f"Gradient directory not found: {gradient_dir}")
        return []
    
    # Find all gradient files
    for filename in os.listdir(gradient_dir):
        if filename.endswith('.pt') and 'round_' in filename and 'client_' in filename:
            filepath = os.path.join(gradient_dir, filename)
            try:
                data = torch.load(filepath)
                
                # Extract round and client from filename
                parts = filename.replace('.pt', '').split('_')
                round_num = int(parts[1])
                client_id = int(parts[3])
                
                # Calculate gradient norm if not already saved
                if 'grad_norm' in data:
                    grad_norm = data['grad_norm']
                else:
                    gradients = data['gradients']
                    grad_norm = torch.stack([g.norm() for g in gradients]).mean().item()
                
                # Vulnerability score = loss * gradient_norm (higher = more vulnerable)
                vulnerability = data['loss'] * grad_norm
                
                vulnerability_scores.append({
                    'round': round_num,
                    'client_id': client_id,
                    'loss': data['loss'],
                    'accuracy': data['accuracy'],
                    'grad_norm': grad_norm,
                    'vulnerability_score': vulnerability,
                    'filename': filepath
                })
                available_files.append(filepath)
                
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
                continue
    
    # Sort by vulnerability score (highest first)
    vulnerability_scores.sort(key=lambda x: x['vulnerability_score'], reverse=True)
    
    return vulnerability_scores

def select_attack_config(vulnerability_score, loss):
    """Select attack configuration based on vulnerability"""
    if vulnerability_score > 5.0 or loss > 3.0:  # High vulnerability
        return SMART_ATTACK_CONFIG, "SMART"
    else:
        return STANDARD_ATTACK_CONFIG, "STANDARD"

def load_federated_data(filepath):
    """Load federated gradient data from file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Gradient file not found: {filepath}")
    
    data = torch.load(filepath)
    return data

def load_ground_truth_from_federated(filepath):
    """Load the original batch from saved gradient data"""
    fed_data = load_federated_data(filepath)
    
    # The framework saves batch_images directly
    ground_truth = fed_data['batch_images']
    labels = fed_data['batch_labels']
    batch_indices = fed_data['batch_indices']
    
    return ground_truth, labels, batch_indices

def get_inversefed_compatible_model(model_arch, num_classes, model_state):
    """Create an inversefed-compatible model or use framework model"""
    # Try to use inversefed models first
    model_mapping = {
        'simplenet': 'ConvNet',
        'convnet': 'ConvNet',
        'mobilenet': 'MobileNet',
        'resnet18': 'ResNet18',
        'shufflenet': 'ShuffleNet'
    }
    
    inversefed_name = model_mapping.get(model_arch.lower(), None)
    
    if inversefed_name:
        try:
            model, _ = inversefed.construct_model(inversefed_name, num_classes=num_classes, num_channels=3)
            model.load_state_dict(model_state)
            return model, inversefed_name
        except:
            pass
    
    # Fall back to framework model
    print(f"Using framework model: {model_arch}")
    model = Net(num_classes=num_classes, arch=model_arch, pretrained=False)
    model.load_state_dict(model_state)
    return model, model_arch

def smart_attack_client(target, save_individual=True):
    """Smart attack on a specific target with optimized config"""
    round_num = target['round']
    client_id = target['client_id']
    vulnerability = target['vulnerability_score']
    
    # Select attack configuration based on vulnerability
    attack_config, config_type = select_attack_config(vulnerability, target['loss'])
    
    print(f"\n{'='*80}")
    print(f"=== SMART ATTACK: Round {round_num}, Client {client_id} ===")
    print(f"Vulnerability Score: {vulnerability:.4f} (Loss: {target['loss']:.4f}, Grad Norm: {target['grad_norm']:.4f})")
    print(f"Attack Config: {config_type}")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load federated data
    print(f"Loading gradient data...")
    fed_data = load_federated_data(target['filename'])
    
    # Get dataset info
    dataset_name = fed_data.get('dataset', cfg.settings['name_dataset']).upper()
    if dataset_name == 'CIFAR10':
        dataset_for_loader = 'CIFAR10'
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif dataset_name == 'CIFAR100':
        dataset_for_loader = 'CIFAR100'
        dm = torch.as_tensor(inversefed.consts.cifar100_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar100_std, **setup)[:, None, None]
    else:
        # Default to CIFAR10 normalization
        dataset_for_loader = 'CIFAR10'
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Load data
    print(f"Loading {dataset_name} dataset...")
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        dataset_for_loader, defs, data_path=f'./datasets/{dataset_name.lower()}'
    )
    
    # Get ground truth from saved data
    print("Loading ground truth images...")
    ground_truth, true_labels, batch_indices = load_ground_truth_from_federated(target['filename'])
    ground_truth = ground_truth.to(**setup)
    true_labels = true_labels.to(device=setup['device'])
    
    # Create model and load the state from federated training
    print(f"Setting up model from round {round_num}, client {client_id}...")
    model_arch = fed_data.get('model_architecture', cfg.settings['arch'])
    num_classes = fed_data.get('num_classes', 10 if dataset_name == 'CIFAR10' else 100)
    
    model, model_name_used = get_inversefed_compatible_model(
        model_arch, num_classes, fed_data['model_state']
    )
    model.to(**setup)
    model.eval()
    
    # Load saved gradients
    saved_gradients = [g.to(**setup) for g in fed_data['gradients']]
    
    print(f"Target Statistics:")
    print(f"  Model: {model_name_used}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Batch indices: {batch_indices[:5]}...")
    print(f"  Classes: {sorted(true_labels.cpu().tolist())}")
    print(f"  Loss: {fed_data['loss']:.4f}")
    print(f"  Accuracy: {fed_data['accuracy']:.2f}%")
    print(f"  Gradient norm: {target['grad_norm']:.4f}")
    print(f"  Vulnerability: {vulnerability:.4f}")
    
    # Print attack configuration
    print(f"\nAttack Configuration ({config_type}):")
    print(f"  Learning rate: {attack_config['lr']}")
    print(f"  Restarts: {attack_config['restarts']}")
    print(f"  Max iterations: {attack_config['max_iterations']}")
    sys.stdout.flush()
    
    print(f"\nStarting smart reconstruction...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Determine number of images to reconstruct
    num_images_to_reconstruct = min(NUM_IMAGES, len(true_labels))
    
    # Perform reconstruction using saved gradients with smart config
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), attack_config, num_images=num_images_to_reconstruct)
    output, stats = rec_machine.reconstruct(saved_gradients, true_labels[:num_images_to_reconstruct], img_shape=(3, 32, 32))
    
    attack_time = time.time() - start_time
    
    # Calculate metrics
    test_mse = (output.detach() - ground_truth[:num_images_to_reconstruct]).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth[:num_images_to_reconstruct], factor=1/ds)
    
    # Per-image PSNR
    per_image_psnr = []
    for i in range(num_images_to_reconstruct):
        psnr = inversefed.metrics.psnr(
            output[i:i+1], 
            ground_truth[i:i+1], 
            factor=1/ds
        )
        per_image_psnr.append(psnr)
    
    # Results
    print(f"\n=== SMART ATTACK RESULTS ===")
    print(f"Round {round_num}, Client {client_id} ({config_type} config)")
    print(f"Vulnerability Score: {vulnerability:.4f}")
    print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    if per_image_psnr:
        print(f"Best image PSNR: {max(per_image_psnr):.2f} dB")
        print(f"Worst image PSNR: {min(per_image_psnr):.2f} dB")
    print(f"MSE: {test_mse:.4f}")
    
    # Quality assessment
    if test_psnr > 25:
        print("🎯 EXCELLENT reconstruction!")
    elif test_psnr > 20:
        print("✅ GOOD reconstruction!")
    elif test_psnr > 15:
        print("⚠️  Fair reconstruction")
    else:
        print("❌ Poor reconstruction")
    
    sys.stdout.flush()
    
    # Save reconstructed images
    output_dir = f'results/gradient_inversion/smart_attack/{cfg.settings["arch"]}_{cfg.settings["name_dataset"]}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save grid with vulnerability info
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    torchvision.utils.save_image(
        output_denorm,
        f'{output_dir}/vuln_{vulnerability:.2f}_round_{round_num}_client_{client_id}_reconstructed.png',
        nrow=5
    )
    
    # Save comparison
    comparison = torch.cat([ground_truth[:num_images_to_reconstruct], output], dim=0)
    comparison_denorm = torch.clamp(comparison * ds + dm, 0, 1)
    torchvision.utils.save_image(
        comparison_denorm,
        f'{output_dir}/vuln_{vulnerability:.2f}_round_{round_num}_client_{client_id}_comparison.png',
        nrow=num_images_to_reconstruct
    )
    
    print(f"Saved to {output_dir}/vuln_{vulnerability:.2f}_round_{round_num}_client_{client_id}_*.png")
    sys.stdout.flush()
    
    return {
        'round': round_num,
        'client_id': client_id,
        'vulnerability_score': vulnerability,
        'config_type': config_type,
        'psnr': test_psnr,
        'mse': test_mse,
        'attack_time': attack_time,
        'gradient_norm': target['grad_norm'],
        'loss': fed_data['loss'],
        'reconstruction_loss': stats['opt'],
        'per_image_psnr': per_image_psnr,
        'batch_indices': batch_indices
    }

def main():
    """Smart federated attack targeting most vulnerable gradients"""
    print("=== SMART FEDERATED GRADIENT ATTACK ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {cfg.settings['name_dataset']}")
    print("Analyzing gradient vulnerability and optimizing attack strategy...")
    sys.stdout.flush()
    
    # Analyze vulnerability of all available gradients
    print("\n=== VULNERABILITY ANALYSIS ===")
    vulnerability_data = analyze_gradient_vulnerability()
    
    if not vulnerability_data:
        print("No gradient files found!")
        print("Please run 'python train_and_attack_single_image.py' first to generate gradients.")
        return
    
    print(f"{'Rank':<5} {'Round':<6} {'Client':<7} {'Loss':<8} {'Acc%':<6} {'Grad Norm':<12} {'Vulnerability':<13} {'Priority':<10}")
    print(f"{'-'*85}")
    
    for i, target in enumerate(vulnerability_data):
        config_type = "SMART" if target['vulnerability_score'] > 5.0 or target['loss'] > 3.0 else "STANDARD"
        print(f"{i+1:<5} {target['round']:<6} {target['client_id']:<7} {target['loss']:<8.4f} "
              f"{target['accuracy']:<6.1f} {target['grad_norm']:<12.4f} {target['vulnerability_score']:<13.4f} {config_type:<10}")
    
    # Attack all targets, prioritizing by vulnerability
    print(f"\n=== SMART ATTACK EXECUTION ===")
    results = []
    
    output_dir = f'results/gradient_inversion/smart_attack/{cfg.settings["arch"]}_{cfg.settings["name_dataset"]}'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, target in enumerate(vulnerability_data):
        try:
            print(f"\n[{i+1}/{len(vulnerability_data)}] Attacking target...")
            result = smart_attack_client(target)
            results.append(result)
            
            # Save intermediate results
            torch.save({
                'results': results,
                'vulnerability_analysis': vulnerability_data,
                'smart_config': SMART_ATTACK_CONFIG,
                'standard_config': STANDARD_ATTACK_CONFIG,
                'model_name': MODEL_NAME,
                'num_images': NUM_IMAGES,
                'method': 'smart_federated_attack',
                'framework': 'cluster-shuffling-fl'
            }, f'{output_dir}/smart_attack_results.pt')
            
        except Exception as e:
            print(f"Error attacking round {target['round']}, client {target['client_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final analysis
    print(f"\n{'='*90}")
    print("=== SMART ATTACK SUMMARY ===")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Round':<6} {'Client':<7} {'Vuln':<10} {'Config':<9} {'PSNR':<8} {'Time(m)':<8} {'Quality':<12}")
    print(f"{'-'*90}")
    
    for i, r in enumerate(results):
        quality = "EXCELLENT" if r['psnr'] > 25 else "GOOD" if r['psnr'] > 20 else "FAIR" if r['psnr'] > 15 else "POOR"
        print(f"{i+1:<5} {r['round']:<6} {r['client_id']:<7} {r['vulnerability_score']:<10.4f} "
              f"{r['config_type']:<9} {r['psnr']:<8.2f} {r['attack_time']/60:<8.1f} {quality:<12}")
    
    if results:
        # Best reconstruction
        best_result = max(results, key=lambda x: x['psnr'])
        print(f"\n🏆 BEST RECONSTRUCTION:")
        print(f"   Round {best_result['round']}, Client {best_result['client_id']}")
        print(f"   Vulnerability: {best_result['vulnerability_score']:.4f}")
        print(f"   PSNR: {best_result['psnr']:.2f} dB")
        print(f"   Config: {best_result['config_type']}")
        
        # Vulnerability vs PSNR correlation
        high_vuln = [r for r in results if r['vulnerability_score'] > 5.0]
        if high_vuln:
            avg_psnr_high = np.mean([r['psnr'] for r in high_vuln])
            low_vuln = [r for r in results if r['vulnerability_score'] <= 5.0]
            if low_vuln:
                avg_psnr_low = np.mean([r['psnr'] for r in low_vuln])
                print(f"\n📊 VULNERABILITY EFFECTIVENESS:")
                print(f"   High vulnerability (>5.0): Avg PSNR = {avg_psnr_high:.2f} dB")
                print(f"   Low vulnerability (≤5.0): Avg PSNR = {avg_psnr_low:.2f} dB")
                print(f"   Smart targeting gain: {avg_psnr_high - avg_psnr_low:.2f} dB")
    
    print(f"\n✅ Smart federated attack complete!")
    print(f"Check '{output_dir}' for results")
    print("Files are named with vulnerability scores for easy identification")
    sys.stdout.flush()

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    main()