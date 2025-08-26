"""
Attack federated learning gradients from main.py architecture
Perform gradient inversion on saved gradients from ModelManager
"""

import torch
import torchvision
import numpy as np
import os
import time
import sys
import glob
import inversefed
from models import Net  # Import your custom models
from config import settings

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_IMAGES = settings['batch_size']  # Use the same batch size as training
ATTACK_ROUNDS = settings['save_gradients_rounds']  # Attack the rounds where gradients were saved

# Attack configuration (same as attack_federated.py)
ATTACK_CONFIG = dict(
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

def find_latest_experiment():
    """Find the latest experiment directory"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        raise FileNotFoundError("No results directory found. Run main.py first!")
    
    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))]
    
    if not exp_dirs:
        raise FileNotFoundError("No experiment directories found in results/")
    
    # Sort by modification time, get most recent
    exp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_exp = exp_dirs[0]
    
    print(f"Using latest experiment: {latest_exp}")
    return os.path.join(results_dir, latest_exp)

def load_gradient_data(exp_dir, round_num, client_id):
    """Load gradient data from ModelManager format"""
    # ModelManager saves gradients as: models/clients/round_XXX/client_id_gradients.pt
    gradient_path = os.path.join(exp_dir, "models", "clients", f"round_{round_num:03d}", f"{client_id}_gradients.pt")
    
    if not os.path.exists(gradient_path):
        raise FileNotFoundError(f"Gradient file not found: {gradient_path}")
    
    print(f"Loading gradients from: {gradient_path}")
    data = torch.load(gradient_path, map_location='cpu')
    return data

def load_model_data(exp_dir, round_num, client_id):
    """Load model data from ModelManager format"""
    model_path = os.path.join(exp_dir, "models", "clients", f"round_{round_num:03d}", f"{client_id}_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data = torch.load(model_path, map_location='cpu')
    return data

def attack_client_gradients(exp_dir, round_num, client_id, save_individual=True):
    """Attack a specific client's gradients"""
    print(f"\n{'='*70}")
    print(f"=== Attacking Round {round_num}, Client {client_id} ===")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # Setup inversefed
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load CIFAR-10 dataset for ground truth reconstruction
    print("Loading CIFAR-10 dataset...")
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        'CIFAR10', defs, data_path='./datasets/cifar10'
    )
    
    # Load gradient data from ModelManager
    print(f"Loading gradient data for round {round_num}, client {client_id}...")
    grad_data = load_gradient_data(exp_dir, round_num, client_id)
    
    # Load model data 
    print(f"Loading model data for round {round_num}, client {client_id}...")
    model_data = load_model_data(exp_dir, round_num, client_id)
    
    # Extract data from ModelManager format
    saved_gradients = grad_data['gradients']
    batch_images = grad_data['batch_images']  # The actual training batch
    batch_labels = grad_data['batch_labels']
    model_state = model_data['model_state']
    loss = grad_data['loss']
    accuracy = grad_data['accuracy']
    
    print(f"Loaded gradient data:")
    print(f"  Gradients: {len(saved_gradients)} tensors")
    print(f"  Batch shape: {batch_images.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Create model with same architecture as training
    print(f"Setting up model...")
    model = Net(num_classes=len(settings.get('classes', range(10))), 
               arch=settings['arch'], 
               pretrained=settings['pretrained'], 
               input_size=(32, 32))
    model.load_state_dict(model_state)
    model.to(**setup)
    model.eval()
    
    # Prepare ground truth data
    ground_truth = batch_images.to(**setup)
    true_labels = batch_labels.to(device=setup['device'])
    
    # Normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Prepare gradients for attack
    gradients_for_attack = [g.to(**setup) for g in saved_gradients]
    
    # Calculate gradient norm
    gradient_norm = torch.stack([g.norm() for g in gradients_for_attack]).mean().item()
    
    print(f"\nGradient Statistics:")
    print(f"  Gradient norm: {gradient_norm:.4f}")
    print(f"  Number of gradient tensors: {len(gradients_for_attack)}")
    
    # Verify gradients by computing fresh ones
    print("Computing fresh gradients for verification...")
    model.zero_grad()
    target_loss_output = loss_fn(model(ground_truth), true_labels)
    # Handle tuple output from loss function (loss, _, _)
    if isinstance(target_loss_output, (tuple, list)):
        target_loss = target_loss_output[0]
    else:
        target_loss = target_loss_output
    fresh_gradients = torch.autograd.grad(target_loss, model.parameters())
    fresh_gradient_norm = torch.stack([g.norm() for g in fresh_gradients]).mean().item()
    print(f"  Fresh gradient norm: {fresh_gradient_norm:.4f}")
    print(f"  Fresh loss: {target_loss.item():.4f}")
    print(f"  Gradient norm ratio: {gradient_norm/fresh_gradient_norm:.4f}")
    sys.stdout.flush()
    
    print(f"\nStarting reconstruction with {ATTACK_CONFIG['restarts']} restarts...")
    print(f"Reconstructing {NUM_IMAGES} images of shape (3, 32, 32)")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Perform gradient inversion attack
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), ATTACK_CONFIG, num_images=NUM_IMAGES)
    output, stats = rec_machine.reconstruct(gradients_for_attack, true_labels, img_shape=(3, 32, 32))
    
    attack_time = time.time() - start_time
    
    # Calculate reconstruction metrics
    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    
    # Per-image PSNR
    per_image_psnr = []
    for i in range(NUM_IMAGES):
        psnr = inversefed.metrics.psnr(
            output[i:i+1], 
            ground_truth[i:i+1], 
            factor=1/ds
        )
        per_image_psnr.append(psnr)
    
    # Print results
    print(f"\n=== Results for Round {round_num}, Client {client_id} ===")
    print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    print(f"Best image PSNR: {max(per_image_psnr):.2f} dB")
    print(f"Worst image PSNR: {min(per_image_psnr):.2f} dB")
    print(f"MSE: {test_mse:.4f}")
    sys.stdout.flush()
    
    # Save reconstructed images
    os.makedirs('main_architecture_reconstructions', exist_ok=True)
    
    # Save grid comparison
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    ground_truth_denorm = torch.clamp(ground_truth * ds + dm, 0, 1)
    
    # Save reconstructions
    torchvision.utils.save_image(
        output_denorm,
        f'main_architecture_reconstructions/round_{round_num}_client_{client_id}_reconstructed.png',
        nrow=5
    )
    
    # Save originals
    torchvision.utils.save_image(
        ground_truth_denorm,
        f'main_architecture_reconstructions/round_{round_num}_client_{client_id}_original.png',
        nrow=5
    )
    
    # Save side-by-side comparison
    comparison = torch.cat([ground_truth_denorm, output_denorm], dim=0)
    torchvision.utils.save_image(
        comparison,
        f'main_architecture_reconstructions/round_{round_num}_client_{client_id}_comparison.png',
        nrow=NUM_IMAGES
    )
    
    print(f"Saved images to main_architecture_reconstructions/round_{round_num}_client_{client_id}_*.png")
    
    # Save individual images if requested
    if save_individual:
        for i in range(NUM_IMAGES):
            # Original
            orig = ground_truth_denorm[i]
            torchvision.utils.save_image(
                orig,
                f'main_architecture_reconstructions/round_{round_num}_client_{client_id}_img_{i}_original.png'
            )
            # Reconstructed
            recon = output_denorm[i]
            torchvision.utils.save_image(
                recon,
                f'main_architecture_reconstructions/round_{round_num}_client_{client_id}_img_{i}_reconstructed.png'
            )
        print(f"Saved individual images for round {round_num}, client {client_id}")
    
    sys.stdout.flush()
    
    return {
        'round': round_num,
        'client_id': client_id,
        'psnr': test_psnr,
        'mse': test_mse,
        'attack_time': attack_time,
        'gradient_norm': gradient_norm,
        'fresh_gradient_norm': fresh_gradient_norm,
        'loss': loss,
        'fresh_loss': target_loss.item(),
        'reconstruction_loss': stats['opt'],
        'per_image_psnr': per_image_psnr,
        'accuracy': accuracy
    }

def main():
    """Attack gradients from main.py federated learning architecture"""
    print("=== Main Architecture Gradient Inversion Attack ===")
    print(f"Device: {DEVICE}")
    print(f"Architecture: {settings['arch']}")
    print(f"Batch size: {NUM_IMAGES}")
    print(f"Attack rounds: {ATTACK_ROUNDS}")
    print(f"Attack configuration:")
    print(f"  Restarts: {ATTACK_CONFIG['restarts']}")
    print(f"  Max iterations: {ATTACK_CONFIG['max_iterations']}")
    print(f"  Total variation: {ATTACK_CONFIG['total_variation']}")
    sys.stdout.flush()
    
    # Find latest experiment
    try:
        exp_dir = find_latest_experiment()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Find all available clients
    clients_dir = os.path.join(exp_dir, "models", "clients")
    available_clients = set()
    
    # Scan for available gradient files
    for round_num in ATTACK_ROUNDS:
        round_dir = os.path.join(clients_dir, f"round_{round_num:03d}")
        if os.path.exists(round_dir):
            for file in os.listdir(round_dir):
                if file.endswith('_gradients.pt'):
                    client_id = file.replace('_gradients.pt', '')
                    available_clients.add(client_id)
    
    available_clients = sorted(list(available_clients))
    print(f"Found clients with saved gradients: {available_clients}")
    
    results = []
    
    # Attack each client in each round
    for round_num in ATTACK_ROUNDS:
        for client_id in available_clients:
            try:
                result = attack_client_gradients(exp_dir, round_num, client_id)
                results.append(result)
                
                # Save intermediate results
                torch.save({
                    'results': results,
                    'attack_config': ATTACK_CONFIG,
                    'architecture': settings['arch'],
                    'num_images': NUM_IMAGES,
                    'method': 'main_architecture_gradients',
                    'experiment_dir': exp_dir
                }, 'main_architecture_reconstructions/attack_results.pt')
                
            except FileNotFoundError as e:
                print(f"Skipping round {round_num}, client {client_id}: {e}")
                continue
            except Exception as e:
                print(f"Error attacking round {round_num}, client {client_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print final summary
    print(f"\n{'='*80}")
    print("=== MAIN ARCHITECTURE ATTACK SUMMARY ===")
    print(f"{'='*80}")
    print(f"{'Round':<6} {'Client':<10} {'PSNR (dB)':<12} {'Time (min)':<12} {'Grad Norm':<12} {'Loss':<8}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['round']:<6} {r['client_id']:<10} {r['psnr']:<12.2f} {r['attack_time']/60:<12.1f} "
              f"{r['gradient_norm']:<12.4f} {r['loss']:<8.4f}")
    
    # Analysis by round
    if len(results) > 0:
        print(f"\n=== Analysis by Round ===")
        for round_num in ATTACK_ROUNDS:
            round_results = [r for r in results if r['round'] == round_num]
            if round_results:
                avg_psnr = np.mean([r['psnr'] for r in round_results])
                print(f"Round {round_num}: {len(round_results)} clients, Avg PSNR = {avg_psnr:.2f} dB")
        
        # Overall statistics
        all_psnr = [r['psnr'] for r in results]
        print(f"\n=== Overall Statistics ===")
        print(f"Total attacks: {len(results)}")
        print(f"Average PSNR: {np.mean(all_psnr):.2f} dB")
        print(f"Best PSNR: {np.max(all_psnr):.2f} dB")
        print(f"Worst PSNR: {np.min(all_psnr):.2f} dB")
        print(f"PSNR std: {np.std(all_psnr):.2f} dB")
        
        # Success rate (PSNR > 20 dB is usually considered good reconstruction)
        good_reconstructions = [r for r in results if r['psnr'] > 20]
        success_rate = len(good_reconstructions) / len(results) * 100
        print(f"Success rate (PSNR > 20 dB): {success_rate:.1f}% ({len(good_reconstructions)}/{len(results)})")
    
    print(f"\n✅ Main architecture attack complete!")
    print("Check 'main_architecture_reconstructions/' for results")
    sys.stdout.flush()

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    main()