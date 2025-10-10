#!/usr/bin/env python3
"""
Federated Learning Gradient Inversion Attack Script
===================================================

Performs gradient inversion attacks on saved gradients from federated learning experiments.
Supports multiple attack strategies:
- Simple gradient inversion (inversefed-style)
- Advanced optimization-based reconstruction
- Multiple restarts for better reconstruction quality

Uses the saved gradients and training artifacts from the results/ folder.

Prerequisites:
    - Run 'python3 main.py' first with save_gradients=True to create training artifacts

Usage:
    # Attack specific experiment
    python3 run_inference_attack.py --experiment ffhq128_classic_c6_r3

    # Attack specific client and round
    python3 run_inference_attack.py --experiment ffhq128_classic_c6_r3 --client c0_1 --round 1

    # Use different attack configurations
    python3 run_inference_attack.py --experiment ffhq128_classic_c6_r3 --config aggressive

    # Attack all clients in a round
    python3 run_inference_attack.py --experiment ffhq128_classic_c6_r3 --round 1 --all-clients

Output:
    - results/[experiment]/attacks/[client]_round_[round]/ - Attack results per client
    - Visual comparisons (PNG)
    - Metrics (JSON)
    - Reconstructed images (PTH)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import local modules
from models.factory import Net

# Note: GIFD implemented using local StyleGAN2 and gifd_core from inference/ directory
from inference.gifd_core.reconstruction_algorithms import GradientReconstructor
from inference.configs import (
    get_federated_gifd_config,
    get_inverting_gradients_config
)
GAN_AVAILABLE = False
Gs = None

# Dataset normalization constants
NORMALIZATION = {
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    },
    'cifar100': {
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    },
    'ffhq128': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'caltech256': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

# Attack configurations
ATTACK_CONFIGS = {
    'quick_test': {
        'num_restarts': 1,
        'max_iterations': 1000,
        'learning_rate': 0.1,
        'total_variation': 1e-3,
        'lr_decay_iterations': [500],
        'lr_decay_factor': 0.5,
        'convergence_threshold': 1e-6,
        'description': 'Quick test with 1 restart'
    },
    'default': {
        'num_restarts': 2,
        'max_iterations': 2000,
        'learning_rate': 0.1,
        'total_variation': 1e-3,
        'lr_decay_iterations': [1000, 1500],
        'lr_decay_factor': 0.5,
        'convergence_threshold': 1e-6,
        'description': 'Balanced attack with 2 restarts'
    },
    'aggressive': {
        'num_restarts': 4,
        'max_iterations': 4000,
        'learning_rate': 0.1,
        'total_variation': 1e-4,
        'lr_decay_iterations': [2000, 3000, 3500],
        'lr_decay_factor': 0.5,
        'convergence_threshold': 1e-7,
        'description': 'Aggressive attack with 4 restarts'
    },
    'high_quality': {
        'num_restarts': 8,
        'max_iterations': 6000,
        'learning_rate': 0.1,
        'total_variation': 1e-4,
        'lr_decay_iterations': [2000, 3500, 4500, 5000],
        'lr_decay_factor': 0.5,
        'convergence_threshold': 1e-8,
        'description': 'Maximum quality with 8 restarts'
    }
}


def load_experiment_config(experiment_path: Path) -> Dict:
    """Load experiment configuration from JSON file"""
    config_path = experiment_path / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config['settings'] if 'settings' in config else config


def find_gradient_files(experiment_path: Path) -> List[Path]:
    """Find all gradient files in the experiment directory"""
    models_dir = experiment_path / 'models' / 'clients'
    gradient_files = []

    if not models_dir.exists():
        return gradient_files

    for round_dir in sorted(models_dir.iterdir()):
        if round_dir.is_dir() and round_dir.name.startswith('round_'):
            for grad_file in round_dir.glob('*_gradients.pt'):
                gradient_files.append(grad_file)

    return gradient_files


def load_gradient_data(gradient_file: Path) -> Dict:
    """Load gradient data from file"""
    print(f"Loading gradient data from {gradient_file.name}...")
    data = torch.load(gradient_file, map_location='cpu')

    # Print summary
    print(f"  Round: {data['round']}")
    print(f"  Client: {data['client_id']}")
    print(f"  Architecture: {data['model_architecture']}")
    print(f"  Dataset: {data['dataset']}")
    print(f"  Loss: {data['loss']:.4f}")
    print(f"  Accuracy: {data['accuracy']:.4f}")
    print(f"  Gradient norm: {data['grad_norm']:.4f}")
    print(f"  Batch size: {data['batch_labels'].shape[0]}")
    print(f"  Image shape: {data['batch_images'].shape}")

    return data


def reconstruct_model(model_architecture: str, num_classes: int, dataset: str, device: torch.device) -> nn.Module:
    """Reconstruct the model from architecture name"""
    print(f"Reconstructing model: {model_architecture} for {dataset} with {num_classes} classes...")

    # Determine input size based on dataset
    input_size_map = {
        'cifar10': (32, 32),
        'cifar100': (32, 32),
        'ffhq128': (128, 128),
        'caltech256': (224, 224)
    }
    input_size = input_size_map.get(dataset, (32, 32))

    # Determine if pretrained
    pretrained = 'resnet' in model_architecture.lower() or 'mobilenet' in model_architecture.lower()

    # Create model
    model = Net(num_classes=num_classes, arch=model_architecture, pretrained=False, input_size=input_size)
    model = model.to(device)
    model.eval()

    print(f"  Model created with input size: {input_size}")

    return model


def replace_maxpool_with_avgpool(model: nn.Module) -> nn.Module:
    """
    Replace MaxPool2d with AvgPool2d to enable gradient inversion with create_graph=True.
    MaxPool2d doesn't support second-order gradients required for the attack.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.MaxPool2d):
            setattr(model, name, nn.AvgPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding
            ))
            print(f"  Replaced {name}: MaxPool2d -> AvgPool2d")
        else:
            replace_maxpool_with_avgpool(module)
    return model


def load_model_weights(model: nn.Module, gradient_data: Dict, device: torch.device):
    """Load model weights from gradient data"""
    if 'model_state' in gradient_data and gradient_data['model_state'] is not None:
        print("Loading model weights from gradient data...")
        model.load_state_dict(gradient_data['model_state'])
        model = model.to(device)
        model.eval()
        print("  Model weights loaded successfully")

        # Replace MaxPool with AvgPool for gradient inversion compatibility
        print("  Replacing MaxPool2d with AvgPool2d for attack compatibility...")
        model = replace_maxpool_with_avgpool(model)
    else:
        print("Warning: No model state found in gradient data, using random initialization")


def run_gan_based_attack(model, gradients, labels, device, attack_config, dm, ds):
    """Run GAN-based GIFD/GIAS attack using gifd_core GradientReconstructor"""

    print("🚀 Running GIFD attack using gifd_core GradientReconstructor...")

    img_shape = (3, 128, 128)  # Hardcode for FFHQ

    # Use config matching the working FFHQ example and DEFAULT_CONFIG
    config = {
        'cost_fn': 'sim_cmpr0',  # From working example
        'optim': 'adam',
        'restarts': attack_config.get('num_restarts', 1),
        'max_iterations': attack_config.get('max_iterations', 1000),
        'lr': attack_config.get('learning_rate', 0.1),
        'total_variation': attack_config.get('total_variation', 1e-3),
        'image_norm': 1e-1,
        'bn_stat': 1e-1,
        'group_lazy': 1e-1,
        'z_norm': 0.0,
        'gifd': True,  # Enable GIFD mode
        'generative_model': 'stylegan2',  # Correct key name from DEFAULT_CONFIG
        'gen_dataset': 'FFHQ64',  # Dataset for generator from DEFAULT_CONFIG
        'init': 'randn',
        'signed': False,
        'indices': 'def',
        'weights': 'equal',
        'steps': [1000],  # Required for GIFD inter-optimizer - steps per layer
        'start_layer': 0,  # Start from first layer
        'end_layer': 8,    # End at layer 8 (default)
        'project': False   # Disable projection for simplicity
    }

    print(f"Using GIFD config: max_iterations={config['max_iterations']}, restarts={config['restarts']}, cost_fn={config['cost_fn']}")

    # Create reconstructor without GAN first, use simpler signature
    reconstructor = GradientReconstructor(
        model,
        device,
        mean_std=(dm, ds),
        config=config,
        num_images=len(labels),
        bn_prior=[]
    )

    # Run reconstruction
    result = reconstructor.reconstruct(
        gradients,
        labels,
        img_shape=img_shape,
        dryrun=False
    )

    print(f"GIFD attack returned type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
    if hasattr(result, '__len__') and len(result) > 0:
        print(f"First element type: {type(result[0])}")

    # Handle different return types like in the working script
    if isinstance(result, tuple) and len(result) == 2:
        output, stats = result
    elif isinstance(result, torch.Tensor):
        output = result
        stats = {'opt': 0.0}  # Default stats
    elif isinstance(result, list) and len(result) > 0:
        # If it's a list, check if elements are tensors or [string, tensor, dict] tuples
        if isinstance(result[0], torch.Tensor):
            # List of tensors
            output = result[0] if len(result) == 1 else torch.stack(result)
        elif isinstance(result[0], list) and len(result[0]) >= 2 and isinstance(result[0][1], torch.Tensor):
            # List of [string, tensor, dict] tuples - extract the tensor (index 1)
            output = result[0][1] if len(result) == 1 else torch.stack([item[1] for item in result])
        else:
            # Fallback: try to stack directly
            output = result[0] if isinstance(result[0], torch.Tensor) else torch.stack(result)
        stats = {'opt': 0.0}  # Default stats
    elif isinstance(result, list) and len(result) == 0:
        # Empty list - attack failed, create dummy output
        print("Warning: GIFD attack returned empty list, falling back to custom gradient inversion")
        dummy_gt = torch.zeros(len(labels), *img_shape, device=device)
        output, stats = run_custom_gradient_inversion(model, gradients, labels, dummy_gt, device, attack_config, dm, ds, 'gradient_inversion')
    else:
        print("Warning: Unexpected return type from GIFD, falling back to custom gradient inversion")
        dummy_gt = torch.zeros(len(labels), *img_shape, device=device)
        output, stats = run_custom_gradient_inversion(model, gradients, labels, dummy_gt, device, attack_config, dm, ds, 'gradient_inversion')

    return output, stats


def run_custom_gradient_inversion(
    model: nn.Module,
    gradients: List[torch.Tensor],
    labels: torch.Tensor,
    ground_truth_images: torch.Tensor,
    device: torch.device,
    attack_config: Dict,
    dm: torch.Tensor,
    ds: torch.Tensor,
    attack_type: str = 'gradient_inversion'
) -> Tuple[torch.Tensor, Dict]:
    global GAN_AVAILABLE, Gs
    """
    Run gradient inversion attack using optimization-based approach

    Args:
        model: Target model
        gradients: List of gradient tensors
        labels: Ground truth labels
        ground_truth_images: Original images (for comparison)
        device: Computing device
        attack_config: Attack configuration
        dm: Dataset mean for denormalization
        ds: Dataset std for denormalization
        attack_type: Type of attack ('gradient_inversion' or 'gifd')

    Returns:
        Tuple of (reconstructed_images, stats)
    """
    print(f"\n🚀 Starting {attack_type.upper()} attack...")
    print(f"   Config: {attack_config.get('description', 'Custom')}")
    print(f"   Restarts: {attack_config['num_restarts']}")
    print(f"   Max iterations: {attack_config['max_iterations']}")

    start_time = time.time()

    # Get image shape from ground truth
    batch_size, num_channels, height, width = ground_truth_images.shape
    img_shape = (num_channels, height, width)

    # Best reconstruction tracking
    best_loss = float('inf')
    best_output = None
    best_restart = 0

    # Run multiple restarts
    for restart in range(attack_config['num_restarts']):
        print(f"\n--- Restart {restart + 1}/{attack_config['num_restarts']} ---")

        # Initialize with GAN-generated images for GIFD or random noise
        if attack_type == 'gifd' and GAN_AVAILABLE:
            print("🎨 Using StyleGAN2 initialization for GIFD")
            latent = torch.randn(batch_size, Gs.latent_size)  # GPU wait, CPU for model
            with torch.no_grad():
                dlatents = Gs.G_mapping(latent)
                dlatents = dlatents.unsqueeze(1).expand(-1, len(Gs), -1)
                gan_images = Gs.G_synthesis(dlatents)
            # GAN images [-1,1] -> [0,1]
            gan_images = (gan_images + 1) / 2
            gan_images = torch.clamp(gan_images, 0, 1)
            # Move to device
            gan_images = gan_images.to(device)
            # Resize to target image shape if needed
            if gan_images.shape[-2:] != img_shape[1:]:
                gan_images = torch.nn.functional.interpolate(gan_images, size=img_shape[1:], mode='bilinear', align_corners=False)
            # Convert to normalized space
            x_trial = (gan_images - dm) / ds
            x_trial = x_trial.detach().requires_grad_(True)
        else:
            x_trial = torch.randn(batch_size, *img_shape, device=device, requires_grad=True)

        # Setup optimizer with learning rate
        optimizer = torch.optim.Adam([x_trial], lr=attack_config['learning_rate'])

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=attack_config.get('lr_decay_iterations', []),
            gamma=attack_config.get('lr_decay_factor', 0.5)
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Track convergence
        prev_loss = float('inf')
        restart_best_loss = float('inf')
        restart_best_output = None

        # Optimization loop
        for iteration in range(attack_config['max_iterations']):
            optimizer.zero_grad()

            # Forward pass through model
            outputs = model(x_trial)
            cls_loss = criterion(outputs, labels)

            # Compute gradients WITH create_graph=True (like the working script)
            # This requires MaxPool2d to be replaced with AvgPool2d
            computed_gradients = torch.autograd.grad(
                cls_loss,
                model.parameters(),
                create_graph=True  # Enable second-order gradients (like working script)
            )

            # Gradient matching loss (cosine similarity - EXACT algorithm from working script)
            grad_loss = 0.0
            for grad_computed, grad_target in zip(computed_gradients, gradients):
                grad_computed_flat = grad_computed.flatten()
                grad_target_flat = grad_target.flatten()

                # Use cosine similarity loss (minimize 1 - cosine_similarity)
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad_computed_flat, grad_target_flat, dim=0
                )
                grad_loss += 1 - cos_sim

            # Total variation regularization
            tv_loss = 0.0
            if attack_config.get('total_variation', 0) > 0:
                tv_loss = (
                    torch.mean(torch.abs(x_trial[:, :, :, :-1] - x_trial[:, :, :, 1:])) +
                    torch.mean(torch.abs(x_trial[:, :, :-1, :] - x_trial[:, :, 1:, :]))
                )

            # Total loss
            total_loss = grad_loss + attack_config.get('total_variation', 0) * tv_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Clamp to valid range (normalized)
            with torch.no_grad():
                x_trial.data = torch.clamp(x_trial.data, -dm/ds, (1-dm)/ds)

            # Track best for this restart
            if total_loss.item() < restart_best_loss:
                restart_best_loss = total_loss.item()
                restart_best_output = x_trial.detach().clone()

            # Print progress
            if iteration % 500 == 0 or iteration == attack_config['max_iterations'] - 1:
                print(f"  Iter {iteration:4d}: Loss = {total_loss.item():.6f} "
                      f"(grad: {grad_loss.item():.6f}, tv: {tv_loss:.6f})")

            # Check convergence
            if abs(prev_loss - total_loss.item()) < attack_config.get('convergence_threshold', 1e-6):
                if iteration > 100:  # Don't stop too early
                    print(f"  Converged at iteration {iteration}")
                    break
            prev_loss = total_loss.item()

        # Update global best
        if restart_best_loss < best_loss:
            best_loss = restart_best_loss
            best_output = restart_best_output
            best_restart = restart + 1
            print(f"  ✓ New best loss: {best_loss:.6f}")

    attack_time = time.time() - start_time

    print(f"\n✓ Attack completed in {attack_time:.2f}s")
    print(f"  Best restart: {best_restart}")
    print(f"  Final loss: {best_loss:.6f}")

    # Compute metrics
    stats = compute_reconstruction_metrics(
        best_output,
        ground_truth_images.to(device),
        model,
        labels,
        dm,
        ds,
        best_loss,
        attack_time
    )

    return best_output, stats


def compute_reconstruction_metrics(
    reconstructed: torch.Tensor,
    ground_truth: torch.Tensor,
    model: nn.Module,
    labels: torch.Tensor,
    dm: torch.Tensor,
    ds: torch.Tensor,
    final_loss: float,
    attack_time: float
) -> Dict:
    """Compute reconstruction quality metrics"""

    # Denormalize images for pixel-space metrics
    recon_denorm = torch.clamp(reconstructed * ds + dm, 0, 1)
    gt_denorm = torch.clamp(ground_truth * ds + dm, 0, 1)

    # Pixel-space metrics
    mse = torch.mean((recon_denorm - gt_denorm) ** 2).item()
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    l1_distance = torch.mean(torch.abs(recon_denorm - gt_denorm)).item()

    # Feature-space metrics
    with torch.no_grad():
        orig_features = model(ground_truth)
        recon_features = model(reconstructed)
        feature_mse = torch.mean((orig_features - recon_features) ** 2).item()

        # Label accuracy
        _, predicted = torch.max(recon_features, 1)
        label_accuracy = (predicted == labels).float().mean().item()

    # SSIM-like metric (simplified)
    ssim_score = 1 - mse  # Simplified, could use actual SSIM

    metrics = {
        'reconstruction_loss': final_loss,
        'mse': mse,
        'psnr': psnr,
        'l1_distance': l1_distance,
        'feature_mse': feature_mse,
        'label_accuracy': label_accuracy,
        'ssim_score': ssim_score,
        'attack_time': attack_time,
        'success': psnr > 15.0  # Threshold for "successful" attack
    }

    return metrics


def plot_reconstruction_comparison(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    labels: torch.Tensor,
    metrics: Dict,
    save_path: Path,
    dataset: str
):
    """Plot comparison between original and reconstructed images"""

    batch_size = original_images.shape[0]

    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    if batch_size == 1:
        axes = axes.reshape(2, 1)

    for i in range(batch_size):
        # Original image
        orig_img = original_images[i].cpu().permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original\nLabel: {labels[i].item()}", fontsize=10)
        axes[0, i].axis('off')

        # Reconstructed image
        recon_img = reconstructed_images[i].cpu().permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f"Reconstructed", fontsize=10)
        axes[1, i].axis('off')

    # Add overall title with metrics
    quality = 'VULNERABLE' if metrics['psnr'] > 25 else 'PROTECTED'
    plt.suptitle(
        f'Gradient Inversion Attack Results - {dataset.upper()}\n'
        f'PSNR: {metrics["psnr"]:.2f} dB | MSE: {metrics["mse"]:.6f} | '
        f'Status: {quality}',
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {save_path}")


def attack_single_gradient(
    gradient_file: Path,
    experiment_config: Dict,
    attack_config: Dict,
    output_dir: Path,
    device: torch.device,
    attack_type: str = 'gradient_inversion'
):
    """Attack a single gradient file"""

    print("\n" + "=" * 80)
    print(f"ATTACKING: {gradient_file.name}")
    print("=" * 80)

    # Load gradient data
    grad_data = load_gradient_data(gradient_file)

    # Extract metadata
    dataset = grad_data['dataset']
    model_arch = grad_data['model_architecture']

    # Infer number of classes from model state (more reliable than batch labels)
    # The batch may not contain all classes (e.g., balanced training with limited samples)
    num_classes = None
    if 'model_state' in grad_data and grad_data['model_state'] is not None:
        # Find the final layer to determine number of classes
        for key in grad_data['model_state'].keys():
            if 'fc.weight' in key or 'classifier.weight' in key or 'linear.weight' in key:
                num_classes = grad_data['model_state'][key].shape[0]
                print(f"  Detected {num_classes} classes from model state")
                break

    # Fallback to batch labels (may be incorrect if not all classes present)
    if num_classes is None:
        num_classes = len(torch.unique(grad_data['batch_labels']))
        print(f"  Warning: Using {num_classes} classes from batch (may be incorrect)")

    # Get normalization constants
    norm = NORMALIZATION.get(dataset, NORMALIZATION['cifar10'])
    dm = torch.tensor(norm['mean'], device=device).view(3, 1, 1)
    ds = torch.tensor(norm['std'], device=device).view(3, 1, 1)

    # Reconstruct model
    model = reconstruct_model(model_arch, num_classes, dataset, device)
    load_model_weights(model, grad_data, device)

    # Prepare attack data
    gradients = [g.to(device) for g in grad_data['gradients']]
    labels = grad_data['batch_labels'].to(device)
    ground_truth = grad_data['batch_images'].to(device)

    # Run attack
    if attack_type in ['gifd', 'gias']:
        reconstructed, stats = run_gan_based_attack(model, gradients, labels, device, attack_config, dm, ds)
    else:
        reconstructed, stats = run_custom_gradient_inversion(
            model, gradients, labels, ground_truth,
            device, attack_config, dm, ds,
            attack_type=attack_type
        )

    # Denormalize for visualization
    recon_denorm = torch.clamp(reconstructed * ds + dm, 0, 1)
    gt_denorm = torch.clamp(ground_truth * ds + dm, 0, 1)

    # Create output directory
    client_id = grad_data['client_id']
    round_num = grad_data['round']
    attack_output = output_dir / f"{client_id}_round_{round_num:03d}"
    attack_output.mkdir(parents=True, exist_ok=True)

    # Save visualizations
    plot_reconstruction_comparison(
        gt_denorm, recon_denorm, labels, stats,
        attack_output / 'reconstruction_comparison.png',
        dataset
    )

    # Save metrics (handle numpy bool)
    metrics_json = {k: float(v) if isinstance(v, (int, float, np.float64, np.float32))
                      else (bool(v) if isinstance(v, np.bool_) else v)
                   for k, v in stats.items()}
    with open(attack_output / 'attack_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # Save reconstructed images
    torch.save({
        'reconstructed': recon_denorm.cpu(),
        'original': gt_denorm.cpu(),
        'labels': labels.cpu(),
        'metrics': stats,
        'gradient_file': str(gradient_file),
        'attack_config': attack_config
    }, attack_output / 'reconstruction_results.pth')

    # Print summary
    print("\n" + "=" * 80)
    print("ATTACK SUMMARY")
    print("=" * 80)
    print(f"Client: {client_id}")
    print(f"Round: {round_num}")
    print(f"Dataset: {dataset}")
    print(f"Architecture: {model_arch}")
    print(f"\nReconstruction Quality:")
    print(f"  PSNR: {stats['psnr']:.2f} dB")
    print(f"  MSE: {stats['mse']:.6f}")
    print(f"  L1 Distance: {stats['l1_distance']:.6f}")
    print(f"  Label Accuracy: {stats['label_accuracy']:.2%}")
    print(f"\nPrivacy Assessment:")
    if stats['psnr'] > 25:
        print(f"  ⚠️  VULNERABLE - High quality reconstruction")
    elif stats['psnr'] > 20:
        print(f"  ⚠️  WEAK PROTECTION - Recognizable reconstruction")
    elif stats['psnr'] > 15:
        print(f"  ✓  MODERATE PROTECTION")
    else:
        print(f"  ✓✓ STRONG PROTECTION")

    print(f"\nResults saved to: {attack_output}")
    print("=" * 80)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Gradient Inversion Attack on Federated Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard gradient inversion attack
  python3 run_inference_attack.py --experiment ffhq128_classic_c6_r3 --client c0_1 --round 1

  # Use different attack configs
  python3 run_inference_attack.py --experiment ffhq128_classic_c6_r3 --round 1 --config aggressive

  # Attack all clients in a round
  python3 run_inference_attack.py --experiment ffhq128_classic_c6_r3 --round 1 --all-clients

Note: For GIFD/GIAS attacks (GAN-based), you need the gifd_core library.
      This script currently implements standard gradient inversion.
        """
    )

    parser.add_argument('--experiment', type=str, default='ffhq128_classic_c6_r3',
                       help='Experiment name (directory in results/)')
    parser.add_argument('--client', type=str, default=None,
                       help='Specific client ID to attack (e.g., c0_1)')
    parser.add_argument('--round', type=int, default=None,
                       help='Specific round to attack')
    parser.add_argument('--all-clients', action='store_true',
                       help='Attack all clients in specified round')
    parser.add_argument('--config', type=str, default='default',
                       choices=list(ATTACK_CONFIGS.keys()),
                       help='Attack configuration preset (default: default)')
    parser.add_argument('--attack-type', type=str, default='gradient_inversion',
                       choices=['gradient_inversion', 'gifd', 'gias'],
                       help='Type of attack algorithm to use (default: gradient_inversion)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, mps, auto)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Load StyleGAN2 generator for GIFD if path exists
    GAN_PATH = Path(__file__).parent / 'inference' / 'gifd_core' / 'genmodels' / 'stylegan2' / 'Gs.pth'
    global GAN_AVAILABLE, Gs
    GAN_AVAILABLE = False
    Gs = None
    if GAN_PATH.exists():
        try:
            from inference.gifd_core.genmodels.stylegan2.models import load
            Gs = load(GAN_PATH, map_location='cpu')
            Gs.eval()
            GAN_AVAILABLE = True
            print("✓ StyleGAN2 generator loaded for GIFD")
        except Exception as e:
            print(f"⚠️ Failed to load StyleGAN2 GAN: {e}")
            GAN_AVAILABLE = False

    print("\n" + "=" * 80)
    print("FEDERATED LEARNING GRADIENT INVERSION ATTACK")
    print("=" * 80)
    print(f"Experiment: {args.experiment}")
    print(f"Attack Type: {args.attack_type.upper()}")
    print(f"Device: {device}")
    print(f"Attack Config: {args.config}")
    print("=" * 80)

    # Warn if GIFD/GIAS selected but not implemented
    if args.attack_type in ['gifd', 'gias']:
        if not GAN_AVAILABLE:
            print("\n⚠️  WARNING: GIFD attacks require StyleGAN2 setup:")
            print("   1. Ensure inference/gifd_core/genmodels/stylegan2/Gs.pth exists")
            print("   2. Use run_gifd_fl.py instead for alternative GIFD support")
            print("\n   Falling back to standard gradient inversion...\n")
            args.attack_type = 'gradient_inversion'
        else:
            print("\n✓ Using GIFD attack with StyleGAN2 initialization")

    # Setup paths
    results_dir = Path('results')
    experiment_path = results_dir / args.experiment

    if not experiment_path.exists():
        print(f"\n❌ Experiment not found: {experiment_path}")
        print(f"\nAvailable experiments:")
        for exp in sorted(results_dir.iterdir()):
            if exp.is_dir():
                print(f"  - {exp.name}")
        return

    # Load experiment configuration
    try:
        experiment_config = load_experiment_config(experiment_path)
        print(f"\n✓ Loaded experiment configuration")
    except Exception as e:
        print(f"\n❌ Failed to load experiment config: {e}")
        return

    # Get attack configuration
    attack_config = ATTACK_CONFIGS[args.config]

    # Find gradient files
    gradient_files = find_gradient_files(experiment_path)

    if not gradient_files:
        print(f"\n❌ No gradient files found in {experiment_path}")
        print(f"   Make sure save_gradients=True in config.py and run training first")
        return

    # Filter gradient files based on arguments
    if args.client:
        gradient_files = [f for f in gradient_files if args.client in f.name]

    if args.round:
        gradient_files = [f for f in gradient_files
                         if f.parent.name == f'round_{args.round:03d}']

    if not gradient_files:
        print(f"\n❌ No gradient files match the criteria")
        return

    print(f"\n✓ Found {len(gradient_files)} gradient file(s) to attack")

    # Create output directory
    output_dir = experiment_path / 'attacks' / f'{args.attack_type}_{args.config}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Attack each gradient file
    all_stats = []
    for i, grad_file in enumerate(gradient_files, 1):
        print(f"\n\nProcessing {i}/{len(gradient_files)}")
        try:
            stats = attack_single_gradient(
                grad_file,
                experiment_config,
                attack_config,
                output_dir,
                device,
                attack_type=args.attack_type
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"\n❌ Attack failed for {grad_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print overall summary
    if all_stats:
        print("\n\n" + "=" * 80)
        print("OVERALL ATTACK SUMMARY")
        print("=" * 80)
        print(f"Total attacks: {len(all_stats)}")
        print(f"Successful: {sum(1 for s in all_stats if s['success'])}")
        print(f"\nAverage Metrics:")
        print(f"  PSNR: {np.mean([s['psnr'] for s in all_stats]):.2f} dB")
        print(f"  MSE: {np.mean([s['mse'] for s in all_stats]):.6f}")
        print(f"  Label Accuracy: {np.mean([s['label_accuracy'] for s in all_stats]):.2%}")
        print(f"  Attack Time: {np.mean([s['attack_time'] for s in all_stats]):.2f}s")
        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Attack interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
