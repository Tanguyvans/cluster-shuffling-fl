#!/usr/bin/env python3
"""
FL FFHQ Gradient Inversion Attack Script
========================================

Performs gradient inversion attacks on FL-trained FFHQ models using saved gradients.
Supports multiple attack types: GIAS, GIFD, and standard gradient inversion.

Usage:
    python attack_fl_ffhq.py --experiment ffhq128_classic_c6_r3 --round 1 --client c0_1 --attack-type gias
    python attack_fl_ffhq.py --experiment ffhq128_classic_c6_r3 --round 2 --client c0_2 --attack-type gifd

Arguments:
    --experiment: Name of experiment directory (e.g., ffhq128_classic_c6_r3)
    --round: FL round to attack (1, 2, 3)
    --client: Client ID to attack (c0_1, c0_2, etc.)
    --attack-type: Attack type (gias, gifd, gradient_inversion)

Output:
    - fl_attack_results.png - Visual comparison
    - fl_attack_results.pth - Full results data
    - fl_attack_metrics.json - Numerical metrics
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
sys.path.append(str(Path(__file__).parent))

# Import gifd_core components
from exploitai.attacks.inference.gifd_core import GradientReconstructor
from exploitai.attacks.inference.gifd_core import metrics
from exploitai.attacks.inference.configs import (
    get_federated_gifd_config,
    get_ffhq_gias_config,
    get_inverting_gradients_config
)

# Import model factory
from models.factory import Net

# FFHQ normalization (ImageNet values matching train_ffhq_resnet.py)
FFHQ_MEAN = [0.485, 0.456, 0.406]
FFHQ_STD = [0.229, 0.224, 0.225]


def load_fl_artifacts(experiment_name, round_num, client_id):
    """
    Load FL training artifacts for gradient inversion attack

    Args:
        experiment_name: Name of experiment (e.g., 'ffhq128_classic_c6_r3')
        round_num: FL round number (1, 2, 3, ...)
        client_id: Client ID (e.g., 'c0_1')

    Returns:
        dict with model, gradients, images, labels, metadata
    """
    results_dir = Path("results") / experiment_name / "models" / "clients" / f"round_{round_num:03d}"

    if not results_dir.exists():
        raise FileNotFoundError(f"FL artifacts not found: {results_dir}")

    print(f"\nLoading FL artifacts from {results_dir}")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - Round: {round_num}")
    print(f"  - Client: {client_id}")

    # Load metadata
    metadata_path = results_dir / f"{client_id}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nMetadata:")
    print(f"  - Architecture: {metadata['model_info']['architecture']}")
    print(f"  - Dataset: {metadata['model_info']['dataset']}")
    print(f"  - Num classes: {metadata['model_info']['num_classes']}")
    print(f"  - Train loss: {metadata['training_metrics']['train_loss']}")
    print(f"  - Train accuracy: {metadata['training_metrics']['train_accuracy']}")

    # Load model weights
    model_path = results_dir / f"{client_id}_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')

    # Create model using Net factory
    # Use CPU only - MPS has compatibility issues with GAN-based attacks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = metadata['model_info']['num_classes']
    architecture = metadata['model_info']['architecture']
    dataset_name = metadata['model_info']['dataset']

    # Determine input size based on dataset
    if 'ffhq' in dataset_name.lower():
        input_size = (128, 128)  # FFHQ uses 128x128
    else:
        input_size = (32, 32)  # CIFAR and others use 32x32

    model = Net(
        num_classes=num_classes,
        arch=architecture,
        pretrained=False,
        input_size=input_size
    )

    # Load model weights - handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        # Assume checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"  - Model loaded: {architecture}")

    # Load gradients
    gradient_path = results_dir / f"{client_id}_gradients.pt"
    gradient_data = torch.load(gradient_path, map_location='cpu')

    gradients = gradient_data['gradients']
    images = gradient_data['batch_images'].to(device)
    labels = gradient_data['batch_labels'].to(device)

    print(f"\nGradient data:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Labels: {labels.tolist()}")
    print(f"  - Num gradient tensors: {len(gradients)}")
    print(f"  - Gradient norm: {gradient_data.get('grad_norm', sum(g.norm().item() for g in gradients)):.4f}")

    return {
        'model': model,
        'gradients': gradients,
        'images': images,
        'labels': labels,
        'metadata': metadata,
        'device': device
    }


def denormalize(tensor, mean=FFHQ_MEAN, std=FFHQ_STD):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def visualize_results(ground_truth, reconstructed, metrics_dict, save_path="fl_attack_results.png"):
    """Visualize attack results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Denormalize for visualization
    gt_img = denormalize(ground_truth.cpu()).permute(1, 2, 0).numpy()
    rec_img = denormalize(reconstructed.cpu()).permute(1, 2, 0).numpy()

    # Clip to valid range
    gt_img = np.clip(gt_img, 0, 1)
    rec_img = np.clip(rec_img, 0, 1)

    # Ground truth
    axes[0, 0].imshow(gt_img)
    axes[0, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Reconstructed
    axes[0, 1].imshow(rec_img)
    axes[0, 1].set_title("Reconstructed", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Difference heatmap
    diff = np.abs(gt_img - rec_img)
    im = axes[1, 0].imshow(diff, cmap='hot')
    axes[1, 0].set_title("Absolute Difference", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Metrics text
    axes[1, 1].axis('off')
    metrics_text = f"""
    Attack Metrics:

    PSNR: {metrics_dict.get('psnr', 0):.2f} dB
    SSIM: {metrics_dict.get('ssim', 0):.4f}
    MSE: {metrics_dict.get('mse', 0):.6f}
    LPIPS: {metrics_dict.get('lpips', 0):.4f}

    Attack Success:
    {"HIGH QUALITY" if metrics_dict.get('psnr', 0) > 25 else
     "MODERATE" if metrics_dict.get('psnr', 0) > 20 else "LOW"}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12,
                    verticalalignment='center', family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to {save_path}")
    plt.close()


def visualize_batch_results(ground_truth_batch, reconstructed_batch, metrics_dict, save_path="fl_attack_batch_results.png"):
    """Visualize batch attack results - show grid of GT vs reconstructed images"""
    num_images = len(ground_truth_batch)

    # Create grid layout: if more than 16 images, use 2 rows of comparisons with multiple columns
    # For 32 images: 4 rows x 16 columns (2 GT rows + 2 reconstructed rows, each showing 16 images)
    if num_images <= 8:
        # Small batch: 2 rows x num_images columns
        ncols = num_images
        fig, axes = plt.subplots(2, ncols, figsize=(2 * ncols, 4))
        if num_images == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_images):
            gt_img = denormalize(ground_truth_batch[i].cpu()).permute(1, 2, 0).numpy()
            rec_img = denormalize(reconstructed_batch[i].cpu()).permute(1, 2, 0).numpy()
            gt_img = np.clip(gt_img, 0, 1)
            rec_img = np.clip(rec_img, 0, 1)

            axes[0, i].imshow(gt_img)
            if i == 0:
                axes[0, i].set_ylabel("GT", fontsize=10, fontweight='bold')
            axes[0, i].set_title(f"{i+1}", fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(rec_img)
            if i == 0:
                axes[1, i].set_ylabel("Rec", fontsize=10, fontweight='bold')
            if 'psnr_per_image' in metrics_dict and i < len(metrics_dict['psnr_per_image']):
                psnr = metrics_dict['psnr_per_image'][i]
                axes[1, i].set_xlabel(f"{psnr:.1f}", fontsize=7)
            axes[1, i].axis('off')
    else:
        # Large batch: split into 2 groups (e.g., 32 images = 2x16)
        ncols = 16
        nrows_per_group = 2  # GT row + reconstructed row
        ngroups = (num_images + ncols - 1) // ncols  # Ceiling division

        fig, axes = plt.subplots(ngroups * nrows_per_group, ncols, figsize=(32, 4 * ngroups))

        for i in range(num_images):
            group = i // ncols  # Which group (0 or 1 for 32 images)
            col = i % ncols     # Column within the group

            gt_img = denormalize(ground_truth_batch[i].cpu()).permute(1, 2, 0).numpy()
            rec_img = denormalize(reconstructed_batch[i].cpu()).permute(1, 2, 0).numpy()
            gt_img = np.clip(gt_img, 0, 1)
            rec_img = np.clip(rec_img, 0, 1)

            # GT row for this group
            gt_row = group * nrows_per_group
            axes[gt_row, col].imshow(gt_img)
            if col == 0:
                axes[gt_row, col].set_ylabel("GT", fontsize=10, fontweight='bold')
            axes[gt_row, col].set_title(f"{i+1}", fontsize=8)
            axes[gt_row, col].axis('off')

            # Reconstructed row for this group
            rec_row = group * nrows_per_group + 1
            axes[rec_row, col].imshow(rec_img)
            if col == 0:
                axes[rec_row, col].set_ylabel("Rec", fontsize=10, fontweight='bold')
            if 'psnr_per_image' in metrics_dict and i < len(metrics_dict['psnr_per_image']):
                psnr = metrics_dict['psnr_per_image'][i]
                axes[rec_row, col].set_xlabel(f"{psnr:.1f}", fontsize=7)
            axes[rec_row, col].axis('off')

        # Hide unused subplots if num_images doesn't fill the grid exactly
        for i in range(num_images, ngroups * ncols):
            group = i // ncols
            col = i % ncols
            gt_row = group * nrows_per_group
            rec_row = group * nrows_per_group + 1
            axes[gt_row, col].axis('off')
            axes[rec_row, col].axis('off')

    # Add overall metrics as suptitle
    avg_psnr = metrics_dict.get('avg_psnr', 0)
    num_total = metrics_dict.get('num_images', num_images)
    fig.suptitle(f"Batch Reconstruction Results - {num_total} images | Avg PSNR: {avg_psnr:.2f} dB",
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBatch results saved to {save_path}")
    plt.close()


def run_attack(experiment_name, round_num, client_id, attack_type='gias'):
    """
    Run gradient inversion attack on FL-trained model

    Args:
        experiment_name: Experiment directory name
        round_num: FL round to attack
        client_id: Client to attack
        attack_type: Attack type (gias, gifd, gradient_inversion)
    """
    print(f"\n{'='*70}")
    print(f"FL FFHQ Gradient Inversion Attack")
    print(f"{'='*70}")

    # Load FL artifacts
    artifacts = load_fl_artifacts(experiment_name, round_num, client_id)

    model = artifacts['model']
    gradients = artifacts['gradients']
    ground_truth_images = artifacts['images']
    ground_truth_labels = artifacts['labels']
    device = artifacts['device']

    # Get attack configuration
    print(f"\nConfiguring {attack_type.upper()} attack...")
    if attack_type == 'gias':
        attack_config = get_ffhq_gias_config()
    elif attack_type == 'gifd':
        attack_config = get_federated_gifd_config()
    else:  # gradient_inversion
        attack_config = get_inverting_gradients_config()

    # Fix config - EXACTLY like attack_ffhq.py lines 270-292
    config = attack_config.copy()

    # Remove keys that are not part of the gifd_core DEFAULT_CONFIG
    keys_to_remove = ['attack_type', 'device', 'learning_rate', 'num_restarts',
                      'inter_optimization', 'verbose', 'generator', 'dataset']
    for key in keys_to_remove:
        if key in config:
            del config[key]

    # Map learning_rate to lr (gifd_core uses 'lr' not 'learning_rate')
    if 'learning_rate' in attack_config:
        config['lr'] = attack_config['learning_rate']

    # Map num_restarts to restarts (gifd_core uses 'restarts' not 'num_restarts')
    if 'num_restarts' in attack_config:
        config['restarts'] = attack_config['num_restarts']

    # Map generator to generative_model (gifd_core uses 'generative_model' not 'generator')
    if 'generator' in attack_config:
        config['generative_model'] = attack_config['generator']

    # Map dataset to gen_dataset (gifd_core uses 'gen_dataset' not 'dataset')
    if 'dataset' in attack_config:
        config['gen_dataset'] = attack_config['dataset']

    # Handle both dict and object configs
    if isinstance(config, dict):
        print(f"  - Optimizer: {config.get('optim', 'N/A')}")
        print(f"  - Cost function: {config.get('cost_fn', 'N/A')}")
        print(f"  - Total iterations: {config.get('max_iterations', 'N/A')}")
        print(f"  - Learning rate: {config.get('learning_rate', config.get('lr', 'N/A'))}")
    else:
        print(f"  - Optimizer: {config.optim}")
        print(f"  - Cost function: {config.cost_fn}")
        print(f"  - Total iterations: {config.max_iterations}")
        print(f"  - Learning rate: {config.lr}")

    # Initialize reconstructor
    dm = torch.tensor(FFHQ_MEAN, device=device).view(3, 1, 1)
    ds = torch.tensor(FFHQ_STD, device=device).view(3, 1, 1)

    # Try to reconstruct the full batch (all images)
    num_images_to_reconstruct = len(ground_truth_labels)

    print(f"\nBatch information:")
    print(f"  - Total images in batch: {len(ground_truth_labels)}")
    print(f"  - Unique classes: {len(torch.unique(ground_truth_labels))} classes")
    print(f"  - Attempting to reconstruct ALL {num_images_to_reconstruct} images from batch")
    print(f"  - Note: This is challenging since multiple images share the same class label")

    reconstructor = GradientReconstructor(
        model,
        device,
        mean_std=(dm, ds),
        config=config,
        num_images=num_images_to_reconstruct,  # Reconstruct all 32 images
        bn_prior=[],
        G=None
    )

    # Run attack
    print(f"\nStarting gradient inversion attack...")
    print(f"  - Target: {client_id} from round {round_num}")
    print(f"  - Reconstructing full batch of {num_images_to_reconstruct} images")
    start_time = time.time()

    # Get image shape from ground truth
    img_shape = tuple(ground_truth_images[0].shape)
    print(f"  - Image shape for reconstruction: {img_shape}")

    # Call reconstruct with positional arguments (like attack_ffhq.py line 307-311)
    reconstructed_data = reconstructor.reconstruct(
        gradients,           # First positional arg
        ground_truth_labels, # Second positional arg
        img_shape=img_shape, # Keyword arg
        dryrun=False         # Keyword arg
    )

    attack_time = time.time() - start_time
    print(f"\nAttack completed in {attack_time:.2f}s")

    # Extract reconstructed image (handle different return types)
    # Debug output to see what we got
    print(f"DEBUG: Reconstructed data type: {type(reconstructed_data)}")
    if isinstance(reconstructed_data, list):
        print(f"DEBUG: List length: {len(reconstructed_data)}")
        if len(reconstructed_data) > 0:
            print(f"DEBUG: First element type: {type(reconstructed_data[0])}")

    if isinstance(reconstructed_data, tuple) and len(reconstructed_data) == 2:
        output, stats = reconstructed_data
    elif isinstance(reconstructed_data, torch.Tensor):
        output = reconstructed_data
    elif isinstance(reconstructed_data, list) and len(reconstructed_data) > 0:
        # Handle list of results (from multiple restarts or iterations)
        # Take the last/best result
        last_result = reconstructed_data[-1] if len(reconstructed_data) > 1 else reconstructed_data[0]

        if isinstance(last_result, torch.Tensor):
            output = last_result
        elif isinstance(last_result, (tuple, list)) and len(last_result) >= 2:
            # Format: (label, tensor, stats) or [label, tensor, stats]
            output = last_result[1] if isinstance(last_result[1], torch.Tensor) else last_result[0]
        else:
            output = last_result
    else:
        raise ValueError(f"Unexpected return type: {type(reconstructed_data)}")

    # Ensure we have the right shape - output should be [num_images, C, H, W]
    if isinstance(output, torch.Tensor):
        if output.dim() == 4:
            reconstructed_imgs = output  # Already [batch, C, H, W]
        elif output.dim() == 3:
            reconstructed_imgs = output.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        else:
            raise ValueError(f"Unexpected tensor shape: {output.shape}")
    else:
        raise ValueError(f"Output is not a tensor: {type(output)}")

    print(f"\nReconstructed {reconstructed_imgs.shape[0]} images")

    # Calculate metrics for each reconstructed image
    print("\nCalculating attack metrics for all images...")

    psnr_list = []
    ssim_list = []
    mse_list = []

    # Calculate metrics for each image pair
    num_to_compare = min(len(ground_truth_images), len(reconstructed_imgs))
    for i in range(num_to_compare):
        gt_img = ground_truth_images[i].cpu().unsqueeze(0)
        rec_img = reconstructed_imgs[i].cpu().unsqueeze(0)

        psnr_val = metrics.psnr(rec_img, gt_img)
        ssim_val = metrics.ssim(rec_img, gt_img)
        mse_val = metrics.total_variation(rec_img - gt_img).item()

        # Convert tensors to floats
        if torch.is_tensor(psnr_val):
            psnr_val = psnr_val.item()
        if torch.is_tensor(ssim_val):
            ssim_val = ssim_val.item()

        psnr_list.append(float(psnr_val))
        ssim_list.append(float(ssim_val))
        mse_list.append(float(mse_val))

    # Calculate average metrics
    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0.0
    avg_mse = sum(mse_list) / len(mse_list) if mse_list else 0.0

    metrics_dict = {
        'avg_psnr': float(avg_psnr),
        'avg_ssim': float(avg_ssim),
        'avg_mse': float(avg_mse),
        'psnr_per_image': psnr_list,
        'ssim_per_image': ssim_list,
        'mse_per_image': mse_list,
        'num_images': num_to_compare,
        'attack_time': float(attack_time),
        'experiment': experiment_name,
        'round': int(round_num),
        'client': client_id,
        'attack_type': attack_type
    }

    print(f"\nAttack Results (averaged over {num_to_compare} images):")
    print(f"  - Average PSNR: {avg_psnr:.2f} dB")
    print(f"  - Average SSIM: {avg_ssim:.4f}")
    print(f"  - Average MSE: {avg_mse:.6f}")
    print(f"  - PSNR range: {min(psnr_list):.2f} - {max(psnr_list):.2f} dB")

    # Visualize and save results - show all images in the batch
    save_prefix = f"fl_{experiment_name}_r{round_num}_{client_id}_{attack_type}"
    num_to_show = num_to_compare  # Show all images

    # Create comparison grid
    visualize_batch_results(
        ground_truth_images[:num_to_show],
        reconstructed_imgs[:num_to_show],
        metrics_dict,
        save_path=f"{save_prefix}_results.png"
    )

    # Save full results
    torch.save({
        'ground_truth': ground_truth_images.cpu(),
        'reconstructed': reconstructed_imgs.cpu(),
        'labels': ground_truth_labels.cpu(),
        'metrics': metrics_dict,
        'config': config if isinstance(config, dict) else config.__dict__
    }, f"{save_prefix}_results.pth")

    # Save metrics JSON
    with open(f"{save_prefix}_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nAll results saved with prefix: {save_prefix}")

    return metrics_dict


def main():
    parser = argparse.ArgumentParser(
        description='FL FFHQ Gradient Inversion Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Attack with default settings (auto-detects latest experiment)
  python3 attack_fl_ffhq.py --attack-type gias

  # Attack specific experiment, round, and client
  python3 attack_fl_ffhq.py --experiment ffhq128_classic_c6_r3 --round 1 --client c0_1 --attack-type gias

  # Attack using direct model path
  python3 attack_fl_ffhq.py --model-path results/ffhq128_classic_c6_r3 --round 2 --client c0_3 --attack-type gifd
        """
    )
    parser.add_argument('--experiment', type=str, default=None,
                      help='Experiment name (directory in results/). Auto-detects if not specified.')
    parser.add_argument('--round', type=int, default=1,
                      help='FL round to attack (default: 1)')
    parser.add_argument('--client', type=str, default='c0_1',
                      help='Client ID to attack (default: c0_1)')
    parser.add_argument('--attack-type', type=str, default='gias',
                      choices=['gias', 'gifd', 'gradient_inversion'],
                      help='Attack type (default: gias)')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Direct path to experiment directory (overrides --experiment)')

    args = parser.parse_args()

    # Handle model-path argument
    if args.model_path:
        model_path = Path(args.model_path)
        if model_path.name.startswith('round_'):
            # User provided round_XXX directory, extract experiment from parent
            args.experiment = model_path.parent.parent.parent.name
            args.round = int(model_path.name.split('_')[1])
            print(f"Detected experiment from path: {args.experiment}, round: {args.round}")
        else:
            # User provided experiment directory
            args.experiment = model_path.name
            print(f"Using experiment from path: {args.experiment}")

    # Auto-detect experiment if not specified
    if args.experiment is None:
        results_dir = Path("results")
        if results_dir.exists():
            # Find the latest ffhq experiment
            experiments = [d for d in results_dir.iterdir() if d.is_dir() and 'ffhq' in d.name.lower()]
            if experiments:
                args.experiment = max(experiments, key=lambda x: x.stat().st_mtime).name
                print(f"Auto-detected experiment: {args.experiment}")
            else:
                print("Error: No FFHQ experiments found in results/")
                print("Available experiments:")
                for d in results_dir.iterdir():
                    if d.is_dir():
                        print(f"  - {d.name}")
                return
        else:
            print("Error: results/ directory not found")
            print("Please run 'python3 main.py' first to train FL models")
            return

    # Run attack
    try:
        metrics_dict = run_attack(
            experiment_name=args.experiment,
            round_num=args.round,
            client_id=args.client,
            attack_type=args.attack_type
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nAvailable experiments:")
        results_dir = Path("results")
        if results_dir.exists():
            for d in results_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return
    except Exception as e:
        print(f"\nError during attack: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print final summary
    print(f"\n{'='*70}")
    print("Attack Summary")
    print(f"{'='*70}")
    print(f"Experiment: {args.experiment}")
    print(f"Round: {args.round}")
    print(f"Client: {args.client}")
    print(f"Attack: {args.attack_type.upper()}")

    # Handle both batch and single-image metrics
    if 'avg_psnr' in metrics_dict:
        # Batch reconstruction
        print(f"Images reconstructed: {metrics_dict['num_images']}")
        print(f"Average PSNR: {metrics_dict['avg_psnr']:.2f} dB")
        print(f"Average SSIM: {metrics_dict['avg_ssim']:.4f}")
        print(f"PSNR range: {min(metrics_dict['psnr_per_image']):.2f} - {max(metrics_dict['psnr_per_image']):.2f} dB")
    else:
        # Single image reconstruction
        print(f"PSNR: {metrics_dict['psnr']:.2f} dB")
        print(f"SSIM: {metrics_dict['ssim']:.4f}")

    print(f"Time: {metrics_dict['attack_time']:.2f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
