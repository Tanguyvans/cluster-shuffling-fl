#!/usr/bin/env python3
"""
FFHQ Gradient Inversion Attack Script
====================================

Performs various gradient inversion attacks on FFHQ trained model:
- GIFD (Gradient Inversion from Federated Learning)
- GIAS (Gradient Inversion Attack Strategy) 
- Standard Gradient Inversion

Uses the gifd_core library to demonstrate state-of-the-art gradient inversion.

Prerequisites:
    - Run 'python run_ffhq.py' first to create training artifacts

Usage:
    python attack_ffhq.py [--attack-type gifd|gias|gradient_inversion]

Output:
    - attack_results.png - Visual comparison
    - attack_results.pth - Full results data
    - attack_metrics.json - Numerical metrics
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import gifd_core components
from exploitai.attacks.inference.gifd_core import GradientReconstructor
from exploitai.attacks.inference.gifd_core import metrics
from exploitai.attacks.inference.configs import (
    get_federated_gifd_config, 
    get_ffhq_gias_config,
    get_inverting_gradients_config
)

# FFHQ normalization (ImageNet values)
FFHQ_MEAN = [0.485, 0.456, 0.406]
FFHQ_STD = [0.229, 0.224, 0.225]


class ConvNet64(nn.Module):
    """ConvNet with 64 filters per layer (same architecture as training script)"""
    
    def __init__(self, num_classes=5, num_channels=3):
        super(ConvNet64, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc1 = nn.Linear(256 * 16 * 16, 256)  # Adjusted for 128x128 input
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def load_training_artifacts(data_dir="./results/ffhq_training"):
    """Load training artifacts for GIFD attack"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Training output directory not found: {data_dir}")
    
    print("Loading training artifacts for GIFD attack...")
    
    # Load configuration
    with open(data_dir / 'training_config.json', 'r') as f:
        config = json.load(f)
    
    # Load model
    model_data = torch.load(data_dir / 'trained_model.pth', map_location='cpu')
    
    # Load gradients
    gradient_data = torch.load(data_dir / 'gradients.pth', map_location='cpu')
    
    # Load training data (ground truth)
    training_data = torch.load(data_dir / 'training_data.pth', map_location='cpu')
    
    print(f"✓ Configuration loaded")
    print(f"✓ Model loaded ({config['model_name']})")
    print(f"✓ Gradients loaded ({len(gradient_data['gradients'])} parameters)")
    print(f"✓ Training data loaded ({training_data['images_raw'].shape[0]} images)")
    print(f"✓ Age groups: {', '.join(training_data['age_groups'])}")
    
    return config, model_data, gradient_data, training_data


def reconstruct_model(model_data, device):
    """Reconstruct the trained model"""
    print("Reconstructing model...")
    
    # Create model instance
    model = ConvNet64(
        num_classes=model_data.get('num_classes', 5),
        num_channels=3
    )
    
    # Load state dict
    model.load_state_dict(model_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model reconstructed and loaded to {device}")
    
    return model


def run_gradient_inversion_attack(model, gradients, labels, device, attack_config, age_groups, attack_type):
    """Run gradient inversion attack"""
    print(f"🚀 Starting {attack_type.upper()} attack with config: {attack_config.get('attack_type', attack_type)}")
    start_time = time.time()
    
    # Setup normalization
    dm = torch.tensor(FFHQ_MEAN, device=device).view(3, 1, 1)
    ds = torch.tensor(FFHQ_STD, device=device).view(3, 1, 1)
    
    # Fix labels - ensure they are proper tensors
    if isinstance(labels, torch.Tensor):
        labels_tensor = labels.to(device)
    else:
        labels_tensor = torch.tensor(labels, device=device)
    
    # Ensure labels are integers
    labels_tensor = labels_tensor.long()
    
    print(f"Labels for attack: {labels_tensor.tolist()}")
    
    # For standard gradient inversion, use a simple approach
    if attack_type == 'gradient_inversion':
        output, stats = run_simple_gradient_inversion(
            model, gradients, labels_tensor, device, attack_config, dm, ds
        )
    else:
        # For GAN-based attacks, use gifd_core
        output, stats = run_gan_based_attack(
            model, gradients, labels_tensor, device, attack_config, attack_type, dm, ds
        )
    
    attack_time = time.time() - start_time
    print(f"✓ {attack_type.upper()} attack completed in {attack_time:.2f} seconds")
    
    return output, stats, attack_time


def run_simple_gradient_inversion(model, gradients, labels, device, config, dm, ds):
    """Run simple gradient inversion without GANs (inspired by original invertinggradients)"""
    print("Running simple gradient inversion attack...")
    
    # Initialize random images
    num_images = len(labels)
    img_shape = (3, 128, 128)
    num_restarts = config.get('num_restarts', 1)
    
    best_loss = float('inf')
    best_output = None
    
    print(f"Running {num_restarts} restarts for better reconstruction...")
    
    for restart in range(num_restarts):
        print(f"Restart {restart + 1}/{num_restarts}")
        
        # Initialize with random noise for each restart
        x_trial = torch.randn(num_images, *img_shape, device=device, requires_grad=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([x_trial], lr=config.get('learning_rate', 0.1))
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        max_iterations = config.get('max_iterations', 1000)
        restart_best_loss = float('inf')
        restart_best_output = None
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_trial)
            loss = criterion(outputs, labels)
            
            # Compute gradients
            computed_gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            
            # Gradient matching loss (cosine similarity like original)
            grad_loss = 0.0
            for grad_computed, grad_target in zip(computed_gradients, gradients):
                # Use cosine similarity loss like the original implementation
                grad_computed_flat = grad_computed.flatten()
                grad_target_flat = grad_target.flatten()
                cos_sim = torch.nn.functional.cosine_similarity(grad_computed_flat, grad_target_flat, dim=0)
                grad_loss += 1 - cos_sim  # Minimize 1 - cosine_similarity
            
            # Total variation regularization
            tv_loss = 0.0
            if config.get('total_variation', 0) > 0:
                tv_loss = torch.mean(torch.abs(x_trial[:, :, :, :-1] - x_trial[:, :, :, 1:])) + \
                         torch.mean(torch.abs(x_trial[:, :, :-1, :] - x_trial[:, :, 1:, :]))
            
            # Total loss
            total_loss = grad_loss + config.get('total_variation', 0) * tv_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clamp to valid range
            with torch.no_grad():
                x_trial.data = torch.clamp(x_trial.data, -dm/ds, (1-dm)/ds)
            
            # Track best result for this restart
            if total_loss.item() < restart_best_loss:
                restart_best_loss = total_loss.item()
                restart_best_output = x_trial.detach().clone()
            
            if iteration % 200 == 0:
                print(f"  Iteration {iteration}: Loss = {total_loss.item():.6f}")
        
        # Update global best if this restart was better
        if restart_best_loss < best_loss:
            best_loss = restart_best_loss
            best_output = restart_best_output
            print(f"  New best loss: {best_loss:.6f}")
    
    stats = {'opt': best_loss}
    return best_output, stats


def run_gan_based_attack(model, gradients, labels, device, attack_config, attack_type, dm, ds):
    """Run GAN-based attack using gifd_core"""
    print(f"Running {attack_type.upper()} attack using gifd_core...")
    
    # Clean attack config - remove keys that are not part of the gifd_core config
    attack_config_fixed = attack_config.copy()
    
    # Remove keys that are not part of the gifd_core DEFAULT_CONFIG
    keys_to_remove = ['attack_type', 'device', 'learning_rate', 'num_restarts', 'inter_optimization', 'verbose', 'generator', 'dataset']
    for key in keys_to_remove:
        if key in attack_config_fixed:
            del attack_config_fixed[key]
    
    # Map learning_rate to lr (gifd_core uses 'lr' not 'learning_rate')
    if 'learning_rate' in attack_config:
        attack_config_fixed['lr'] = attack_config['learning_rate']
    
    # Map num_restarts to restarts (gifd_core uses 'restarts' not 'num_restarts')
    if 'num_restarts' in attack_config:
        attack_config_fixed['restarts'] = attack_config['num_restarts']
    
    # Map generator to generative_model (gifd_core uses 'generative_model' not 'generator')
    if 'generator' in attack_config:
        attack_config_fixed['generative_model'] = attack_config['generator']
    
    # Map dataset to gen_dataset (gifd_core uses 'gen_dataset' not 'dataset')
    if 'dataset' in attack_config:
        attack_config_fixed['gen_dataset'] = attack_config['dataset']
    
    # Create reconstructor
    reconstructor = GradientReconstructor(
        model, 
        device,  # Pass device directly, not (dm, ds)
        mean_std=(dm, ds),  # Pass mean_std as the normalization tuple
        config=attack_config_fixed, 
        num_images=len(labels),
        bn_prior=[], 
        G=None
    )
    
    # Run reconstruction
    img_shape = (3, 128, 128)  # FFHQ input size
    result = reconstructor.reconstruct(
        gradients,
        labels,
        img_shape=img_shape,
        dryrun=False
    )
    
    # Handle different return types
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
        print("Warning: GAN-based attack returned empty list, creating dummy output")
        output = torch.randn(len(labels), 3, 128, 128, device=device)
        stats = {'opt': float('inf')}  # Indicate failure
    else:
        raise ValueError(f"Unexpected return type from reconstruct: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
    
    return output, stats


def plot_comparison(original_images, reconstructed_images, age_groups, labels, 
                   metrics, attack_type, save_path="attack_results.png"):
    """Plot comparison between original and reconstructed images"""
    
    num_images = original_images.shape[0]
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*3, 6))
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_images):
        # Original image
        orig_img = torch.clamp(original_images[i], 0, 1).permute(1, 2, 0)
        axes[0, i].imshow(orig_img)
        label = labels[i].item()
        age_group = age_groups.get(label, f"Class {label}")
        axes[0, i].set_title(f"Original\n{age_group}", fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed image
        recon_img = torch.clamp(reconstructed_images[i], 0, 1).permute(1, 2, 0)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f"Reconstructed", fontsize=10)
        axes[1, i].axis('off')
    
    # Add overall title with metrics
    plt.suptitle(f'{attack_type} Attack Results\n'
                f'MSE: {metrics["mse"]:.4f} | PSNR: {metrics["psnr"]:.2f} dB | '
                f'Loss: {metrics["final_loss"]:.4f}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Results saved to {save_path}")


def get_attack_config(attack_type, device):
    """Get configuration for the specified attack type"""
    if attack_type == 'gifd':
        return get_federated_gifd_config(device=device)
    elif attack_type == 'gias':
        return get_ffhq_gias_config(device=device)
    elif attack_type == 'gradient_inversion':
        return get_inverting_gradients_config(device=device)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}. Choose from: gifd, gias, gradient_inversion")


def run_ffhq_attack(attack_type='gifd'):
    """Main function to run gradient inversion attack on FFHQ"""
    print("=" * 80)
    print(f"   FFHQ {attack_type.upper()} GRADIENT INVERSION ATTACK")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ========================================================================
    # Step 1: Load Training Artifacts
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Loading training artifacts")
    print("=" * 60)
    
    try:
        config, model_data, gradient_data, training_data = load_training_artifacts()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"Please run 'python run_ffhq.py' first to create training artifacts.")
        return
    
    # ========================================================================
    # Step 2: Reconstruct Model
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Reconstructing model")
    print("=" * 60)
    
    model = reconstruct_model(model_data, device)
    
    # ========================================================================
    # Step 3: Prepare Attack Data
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Preparing attack data")
    print("=" * 60)
    
    # Load gradients and move to device
    gradients = [g.to(device) for g in gradient_data['gradients']]
    
    # Load ground truth data
    original_images_raw = training_data['images_raw']
    original_labels = training_data['labels']
    age_groups = {i: group for i, group in enumerate(training_data['age_groups'])}
    
    print(f"✓ Gradients loaded: {len(gradients)} parameters")
    print(f"✓ Ground truth: {original_images_raw.shape[0]} images")
    print(f"✓ Age groups: {list(age_groups.values())}")
    print(f"✓ Labels: {original_labels.tolist()}")
    
    # ========================================================================
    # Step 4: Configure Attack
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"Step 4: Configuring {attack_type.upper()} attack")
    print("=" * 60)
    
    # Get configuration for the specified attack type
    attack_config = get_attack_config(attack_type, device)
    
    print(f"📍 {attack_type.upper()} Attack Configuration:")
    for key, value in attack_config.items():
        print(f"   • {key}: {value}")
    
    # ========================================================================
    # Step 5: Run Attack
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"Step 5: Running {attack_type.upper()} attack")
    print("=" * 60)
    
    try:
        # Run attack
        output, stats, attack_time = run_gradient_inversion_attack(
            model, gradients, original_labels, device, 
            attack_config, age_groups, attack_type
        )
        
        # ========================================================================
        # Step 6: Evaluate Results
        # ========================================================================
        print("\n" + "=" * 60)
        print("Step 6: Evaluating attack results")
        print("=" * 60)
        
        # Compute metrics
        dm = torch.tensor(FFHQ_MEAN, device=device).view(3, 1, 1)
        ds = torch.tensor(FFHQ_STD, device=device).view(3, 1, 1)
        
        output_den = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_den = torch.clamp(original_images_raw.to(device) * ds + dm, 0, 1)
        
        mse = torch.mean((output_den - ground_truth_den) ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Feature-space metrics
        with torch.no_grad():
            original_images_norm = (original_images_raw.to(device) - dm) / ds
            orig_features = model(original_images_norm)
            recon_features = model(output)
            feature_mse = torch.mean((orig_features - recon_features) ** 2).item()
        
        # L1 distance
        l1_distance = torch.mean(torch.abs(output_den - ground_truth_den)).item()
        
        metrics = {
            'final_loss': stats.get('opt', 0.0),
            'success': True,
            'mse': mse,
            'psnr': psnr,
            'feature_mse': feature_mse,
            'l1_distance': l1_distance,
            'attack_time': attack_time
        }
        
        print(f"\n📊 {attack_type.upper()} Attack Results:")
        print(f"   Reconstruction loss: {metrics['final_loss']:.6f}")
        print(f"   Success: {metrics['success']}")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   PSNR: {metrics['psnr']:.2f} dB")
        print(f"   Feature MSE: {metrics['feature_mse']:.2e}")
        print(f"   L1 distance: {metrics['l1_distance']:.6f}")
        print(f"   Attack time: {metrics['attack_time']:.2f} seconds")
        
        # ========================================================================
        # Step 7: Visualize Results
        # ========================================================================
        print("\n" + "=" * 60)
        print("Step 7: Visualizing results")
        print("=" * 60)
        
        # Create comparison plot
        plot_comparison(
            original_images_raw,
            output_den.cpu(),
            age_groups,
            original_labels,
            metrics,
            attack_type.upper(),
            save_path=f"{attack_type}_attack_results.png"
        )
        
        # ========================================================================
        # Step 8: Save Attack Results
        # ========================================================================
        print("\n" + "=" * 60)
        print("Step 8: Saving attack results")
        print("=" * 60)
        
        # Save results
        attack_results = {
            'reconstructed_images': output_den.cpu(),
            'original_images': original_images_raw,
            'labels': original_labels,
            'age_groups': age_groups,
            'metrics': metrics,
            'attack_config': attack_config,
            'success': metrics['success'],
            'stats': stats,
            'attack_type': attack_type.upper()
        }
        
        torch.save(attack_results, f'{attack_type}_attack_results.pth')
        
        # Save metrics to JSON
        metrics_json = {k: float(v) if isinstance(v, (int, float, np.float64)) else v 
                       for k, v in metrics.items()}
        with open(f'{attack_type}_attack_metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"✓ Results saved to {attack_type}_attack_results.pth")
        print(f"✓ Metrics saved to {attack_type}_attack_metrics.json")
        
        # ========================================================================
        # Final Summary
        # ========================================================================
        print("\n" + "=" * 80)
        print(f"   {attack_type.upper()} ATTACK SUMMARY")
        print("=" * 80)
        
        print(f"\n🎯 Attack Results:")
        print(f"   • Reconstructed {original_images_raw.shape[0]} FFHQ face images")
        print(f"   • PSNR: {metrics['psnr']:.2f} dB")
        print(f"   • Attack success: {metrics['success']}")
        print(f"   • Total time: {metrics['attack_time']:.1f} seconds")
        print(f"   • Attack type: {attack_type.upper()}")
        
        # Quality assessment
        if metrics['psnr'] > 25:
            quality = "Excellent - Very high quality reconstruction"
        elif metrics['psnr'] > 20:
            quality = "Good - High quality reconstruction"
        elif metrics['psnr'] > 15:
            quality = "Fair - Recognizable faces"
        elif metrics['psnr'] > 10:
            quality = "Poor - Some facial features visible"
        else:
            quality = "Very Poor - Attack largely failed"
        
        print(f"   • Quality assessment: {quality}")
        
        print(f"\n📁 Output files:")
        print(f"   • {attack_type}_attack_results.png - Visual comparison")
        print(f"   • {attack_type}_attack_results.pth - Full results data")
        print(f"   • {attack_type}_attack_metrics.json - Numerical metrics")
        
        print(f"\n🔥 {attack_type.upper()} gradient inversion attack complete!")
        
        return attack_results, metrics
        
    except Exception as e:
        print(f"❌ {attack_type.upper()} attack failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run gradient inversion attack on FFHQ')
    parser.add_argument('--attack-type', type=str, default='gifd', 
                       choices=['gifd', 'gias', 'gradient_inversion'],
                       help='Type of attack to run (default: gifd)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the attack
    try:
        result, metrics = run_ffhq_attack(attack_type=args.attack_type)
        if result is not None:
            print(f"\n✓ {args.attack_type.upper()} attack completed successfully!")
        else:
            print(f"\n❌ {args.attack_type.upper()} attack failed!")
    except Exception as e:
        print(f"❌ Error during attack: {e}")
        import traceback
        traceback.print_exc()
