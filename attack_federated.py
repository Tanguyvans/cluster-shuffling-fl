"""
Attack federated learning gradients from both clients
Perform gradient inversion on saved federated gradients
"""

import torch
import torchvision
import numpy as np
import os
import time
import sys
import inversefed
from models import Net  # Import your custom models

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'ConvNet'
NUM_IMAGES = 10
NUM_CLIENTS = 2

# Attack configuration (same as attack.py)
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

def load_federated_data(round_num, client_id):
    """Load federated gradient data for specific round and client"""
    filename = f'federated_simple_gradients/round_{round_num}_client_{client_id}.pt'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Gradient file not found: {filename}")
    
    data = torch.load(filename)
    return data

def load_ground_truth_from_federated(round_num, client_id, validloader):
    """Load the original batch used by the federated client"""
    fed_data = load_federated_data(round_num, client_id)
    batch_indices = fed_data['batch_indices']
    
    images = []
    labels = []
    for idx in batch_indices:
        img, label = validloader.dataset[idx]
        images.append(img)
        labels.append(label)
    
    ground_truth = torch.stack(images)
    labels = torch.tensor(labels)
    
    return ground_truth, labels, batch_indices

def attack_federated_client(round_num, client_id, save_individual=True):
    """Attack a specific federated client's gradients"""
    print(f"\n{'='*70}")
    print(f"=== Attacking Round {round_num}, Client {client_id} ===")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        'CIFAR10', defs, data_path='./datasets/cifar10'
    )
    
    # Load federated data
    print(f"Loading federated data for round {round_num}, client {client_id}...")
    fed_data = load_federated_data(round_num, client_id)
    
    # Get ground truth for this client
    print("Loading ground truth images...")
    ground_truth, true_labels, batch_indices = load_ground_truth_from_federated(
        round_num, client_id, validloader
    )
    ground_truth = ground_truth.to(**setup)
    true_labels = true_labels.to(device=setup['device'])
    
    # Create model and load the state from federated training
    print(f"Setting up model from round {round_num}, client {client_id}...")
    model = Net(num_classes=10, arch='convnet', pretrained=False, input_size=(32, 32))
    model.load_state_dict(fed_data['model_state'])
    model.to(**setup)
    model.eval()
    
    # Normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Load saved gradients
    saved_gradients = [g.to(**setup) for g in fed_data['gradients']]
    
    # Calculate gradient norm
    gradient_norm = torch.stack([g.norm() for g in saved_gradients]).mean().item()
    
    print(f"Round {round_num}, Client {client_id} Statistics:")
    print(f"  Batch indices: {batch_indices}")
    print(f"  Classes: {sorted(true_labels.cpu().tolist())}")
    print(f"  Loss: {fed_data['loss']:.4f}")
    print(f"  Accuracy: {fed_data['accuracy']:.2f}%")
    print(f"  Gradient norm: {gradient_norm:.4f}")
    
    # Verify gradients by computing fresh ones
    print("Computing fresh gradients for verification...")
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), true_labels)
    fresh_gradients = torch.autograd.grad(target_loss, model.parameters())
    fresh_gradient_norm = torch.stack([g.norm() for g in fresh_gradients]).mean().item()
    print(f"  Fresh gradient norm: {fresh_gradient_norm:.4f}")
    print(f"  Fresh loss: {target_loss.item():.4f}")
    sys.stdout.flush()
    
    print(f"\nStarting reconstruction with {ATTACK_CONFIG['restarts']} restarts...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Perform reconstruction using saved gradients
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), ATTACK_CONFIG, num_images=NUM_IMAGES)
    output, stats = rec_machine.reconstruct(saved_gradients, true_labels, img_shape=(3, 32, 32))
    
    attack_time = time.time() - start_time
    
    # Calculate metrics
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
    
    # Results
    print(f"\n=== Results for Round {round_num}, Client {client_id} ===")
    print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    print(f"Best image PSNR: {max(per_image_psnr):.2f} dB")
    print(f"Worst image PSNR: {min(per_image_psnr):.2f} dB")
    print(f"MSE: {test_mse:.4f}")
    sys.stdout.flush()
    
    # Save reconstructed images
    os.makedirs('federated_reconstructions', exist_ok=True)
    
    # Save grid
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    torchvision.utils.save_image(
        output_denorm,
        f'federated_reconstructions/round_{round_num}_client_{client_id}_reconstructed.png',
        nrow=5
    )
    
    # Save comparison (original on top, reconstructed on bottom)
    comparison = torch.cat([ground_truth, output], dim=0)
    comparison_denorm = torch.clamp(comparison * ds + dm, 0, 1)
    torchvision.utils.save_image(
        comparison_denorm,
        f'federated_reconstructions/round_{round_num}_client_{client_id}_comparison.png',
        nrow=10
    )
    
    print(f"Saved images to federated_reconstructions/round_{round_num}_client_{client_id}_*.png")
    
    # Save individual images if requested
    if save_individual:
        for i in range(NUM_IMAGES):
            # Original
            orig = torch.clamp(ground_truth[i] * ds + dm, 0, 1)
            torchvision.utils.save_image(
                orig,
                f'federated_reconstructions/round_{round_num}_client_{client_id}_img_{i}_original.png'
            )
            # Reconstructed
            recon = torch.clamp(output[i] * ds + dm, 0, 1)
            torchvision.utils.save_image(
                recon,
                f'federated_reconstructions/round_{round_num}_client_{client_id}_img_{i}_reconstructed.png'
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
        'loss': fed_data['loss'],
        'fresh_loss': target_loss.item(),
        'reconstruction_loss': stats['opt'],
        'per_image_psnr': per_image_psnr,
        'batch_indices': batch_indices
    }

def main():
    """Attack federated gradients for both clients at rounds 1, 5, 10"""
    print("=== Federated Gradient Inversion Attack ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Attacking both clients at rounds 1, 5, and 10")
    print(f"Attack configuration:")
    print(f"  Restarts: {ATTACK_CONFIG['restarts']}")
    print(f"  Max iterations: {ATTACK_CONFIG['max_iterations']}")
    print(f"  Total variation: {ATTACK_CONFIG['total_variation']}")
    sys.stdout.flush()
    
    # Attack combinations: (round, client)
    attack_targets = [
        (1, 0), (1, 1),  # Round 1: both clients
        (5, 0), (5, 1),  # Round 5: both clients
        (10, 0), (10, 1) # Round 10: both clients
    ]
    
    results = []
    
    for round_num, client_id in attack_targets:
        try:
            result = attack_federated_client(round_num, client_id)
            results.append(result)
            
            # Save intermediate results
            torch.save({
                'results': results,
                'attack_config': ATTACK_CONFIG,
                'model_name': MODEL_NAME,
                'num_images': NUM_IMAGES,
                'method': 'federated_gradients'
            }, 'federated_reconstructions/federated_attack_results.pt')
            
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
    print("=== FEDERATED ATTACK SUMMARY ===")
    print(f"{'='*80}")
    print(f"{'Round':<6} {'Client':<7} {'PSNR (dB)':<12} {'Time (min)':<12} {'Grad Norm':<12} {'Loss':<8}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['round']:<6} {r['client_id']:<7} {r['psnr']:<12.2f} {r['attack_time']/60:<12.1f} "
              f"{r['gradient_norm']:<12.4f} {r['loss']:<8.4f}")
    
    # Analysis by round
    if len(results) > 0:
        print(f"\n=== Analysis by Round ===")
        for round_num in [1, 5, 10]:
            round_results = [r for r in results if r['round'] == round_num]
            if round_results:
                avg_psnr = np.mean([r['psnr'] for r in round_results])
                print(f"Round {round_num}: Avg PSNR = {avg_psnr:.2f} dB")
        
        # Early vs late comparison
        early_results = [r for r in results if r['round'] == 1]
        late_results = [r for r in results if r['round'] == 10]
        
        if early_results and late_results:
            early_psnr = np.mean([r['psnr'] for r in early_results])
            late_psnr = np.mean([r['psnr'] for r in late_results])
            print(f"\nEarly round (1) avg PSNR: {early_psnr:.2f} dB")
            print(f"Late round (10) avg PSNR: {late_psnr:.2f} dB")
            print(f"Difference: {early_psnr - late_psnr:.2f} dB")
            
            if early_psnr - late_psnr > 3:
                print("✅ Early rounds are more vulnerable!")
            else:
                print("⚠️  Similar vulnerability across rounds")
    
    print(f"\n✅ Federated attack complete!")
    print("Check 'federated_reconstructions/' for results")
    sys.stdout.flush()

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    main()