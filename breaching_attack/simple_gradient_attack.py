import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class SimpleNet(nn.Module):
    """SimpleNet architecture matching your trained models"""
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_trained_model(model_path, device):
    """Load a trained model from .npz file"""
    print(f"Loading model from {model_path}")
    
    # Load the .npz file
    model_data = np.load(model_path)
    print(f"Available keys in model file: {list(model_data.keys())}")
    
    # Create SimpleNet model
    model = SimpleNet(num_classes=10)
    
    # Convert numpy arrays to torch tensors and load into model
    state_dict = {}
    
    # Check if it's the new individual client model format (param_0, param_1, etc.)
    if 'param_0' in model_data:
        print("Detected individual client model format (param_0, param_1, ...)")
        # Individual client models save parameters as param_0, param_1, etc.
        # We need to map these to the correct layer names
        param_names = [
            'conv1.weight', 'conv1.bias',
            'conv2.weight', 'conv2.bias', 
            'fc1.weight', 'fc1.bias',
            'fc2.weight', 'fc2.bias',
            'fc3.weight', 'fc3.bias'
        ]
        
        for i, param_name in enumerate(param_names):
            param_key = f'param_{i}'
            if param_key in model_data:
                state_dict[param_name] = torch.from_numpy(model_data[param_key])
                print(f"Loaded {param_name}: {model_data[param_key].shape}")
            else:
                print(f"Warning: {param_key} not found in model data")
    
    else:
        print("Detected global/cluster model format (model.layer.param)")
        # Map the parameter names from the saved model to PyTorch names
        param_mapping = {
            'model.conv1.weight': 'conv1.weight',
            'model.conv1.bias': 'conv1.bias', 
            'model.conv2.weight': 'conv2.weight',
            'model.conv2.bias': 'conv2.bias',
            'model.fc1.weight': 'fc1.weight',
            'model.fc1.bias': 'fc1.bias',
            'model.fc2.weight': 'fc2.weight', 
            'model.fc2.bias': 'fc2.bias',
            'model.fc3.weight': 'fc3.weight',
            'model.fc3.bias': 'fc3.bias'
        }
        
        for saved_name, pytorch_name in param_mapping.items():
            if saved_name in model_data:
                state_dict[pytorch_name] = torch.from_numpy(model_data[saved_name])
                print(f"Loaded {pytorch_name}: {model_data[saved_name].shape}")
    
    if not state_dict:
        raise ValueError(f"Could not load any parameters from {model_path}. Available keys: {list(model_data.keys())}")
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Enable gradients for model parameters (needed for gradient computation in attack)
    for param in model.parameters():
        param.requires_grad = True
    
    model.train()  # Set to training mode for gradient computation
    
    print("Model loaded successfully!")
    return model

def get_real_data_batch(batch_size=8):
    """Get a batch of real CIFAR-10 data for the attack"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get one batch
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    return images, labels

def compute_gradients(model, images, labels, loss_fn):
    """Compute gradients for a batch of data"""
    model.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    gradients = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return [grad.detach().clone() for grad in gradients], loss

def gradient_inversion_attack(model, target_gradients, batch_size, num_classes=10, num_iterations=5000, lr=0.1):
    """
    Gradient inversion attack to reconstruct images from gradients
    """
    device = next(model.parameters()).device
    
    # Initialize dummy data and labels
    dummy_images = torch.randn((batch_size, 3, 32, 32), device=device, requires_grad=True)
    dummy_labels = torch.randint(0, num_classes, (batch_size,), device=device, requires_grad=False)
    
    # Optimizer for dummy data - using Adam instead of LBFGS to avoid graph issues
    optimizer = optim.Adam([dummy_images], lr=lr)
    
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Starting gradient inversion attack...")
    print(f"Target batch size: {batch_size}")
    print(f"Number of target gradients: {len(target_gradients)}")
    
    history = []
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass with dummy data
        dummy_outputs = model(dummy_images)
        dummy_loss = loss_fn(dummy_outputs, dummy_labels)
        
        # Compute gradients of dummy data
        dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        
        # Compute gradient matching loss
        grad_loss = 0
        for grad_dummy, grad_target in zip(dummy_gradients, target_gradients):
            grad_loss += ((grad_dummy - grad_target) ** 2).sum()
        
        # Add image priors/regularization
        # Total variation loss for smoothness
        tv_loss = torch.sum(torch.abs(dummy_images[:, :, :, :-1] - dummy_images[:, :, :, 1:])) + \
                 torch.sum(torch.abs(dummy_images[:, :, :-1, :] - dummy_images[:, :, 1:, :]))
        
        # L2 regularization
        l2_loss = torch.sum(dummy_images ** 2)
        
        total_loss = grad_loss + 0.01 * tv_loss + 0.0001 * l2_loss
        total_loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Total Loss = {total_loss.item():.6f}")
            history.append(total_loss.item())
    
    return dummy_images.detach(), history

def denormalize_images(images):
    """Denormalize CIFAR-10 images for visualization"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    
    if images.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    denorm_images = images * std + mean
    return torch.clamp(denorm_images, 0, 1)

def plot_comparison(original_images, reconstructed_images, output_path):
    """Plot original vs reconstructed images"""
    batch_size = original_images.shape[0]
    
    # For large batch sizes, create a grid layout
    if batch_size <= 8:
        # Single row layout for small batches
        cols = batch_size
        rows = 2  # Original and reconstructed
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 4))
    else:
        # Grid layout for larger batches
        cols = min(8, batch_size)  # Max 8 columns
        rows = 2 * ((batch_size + cols - 1) // cols)  # 2 rows per image row (orig + recon)
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    
    # Denormalize for visualization
    orig_denorm = denormalize_images(original_images)
    recon_denorm = denormalize_images(reconstructed_images)
    
    # Handle different subplot layouts
    if batch_size == 1:
        # Special case for single image
        axes[0].imshow(orig_denorm[0].cpu().permute(1, 2, 0))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(recon_denorm[0].cpu().permute(1, 2, 0))
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
    elif batch_size <= 8:
        # Single row layout
        for i in range(batch_size):
            axes[0, i].imshow(orig_denorm[i].cpu().permute(1, 2, 0))
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon_denorm[i].cpu().permute(1, 2, 0))
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        # Hide unused subplots
        for i in range(batch_size, cols):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
    else:
        # Grid layout for larger batches
        for i in range(batch_size):
            # Calculate position in grid
            img_row = i // cols
            img_col = i % cols
            
            # Original image goes in even rows (0, 2, 4, ...)
            orig_row = img_row * 2
            axes[orig_row, img_col].imshow(orig_denorm[i].cpu().permute(1, 2, 0))
            axes[orig_row, img_col].set_title(f'Orig {i+1}', fontsize=8)
            axes[orig_row, img_col].axis('off')
            
            # Reconstructed image goes in odd rows (1, 3, 5, ...)
            recon_row = img_row * 2 + 1
            axes[recon_row, img_col].imshow(recon_denorm[i].cpu().permute(1, 2, 0))
            axes[recon_row, img_col].set_title(f'Recon {i+1}', fontsize=8)
            axes[recon_row, img_col].axis('off')
        
        # Hide unused subplots
        total_img_rows = (batch_size + cols - 1) // cols
        for row in range(total_img_rows * 2):
            for col in range(cols):
                if row % 2 == 0:  # Original image row
                    img_idx = (row // 2) * cols + col
                    if img_idx >= batch_size:
                        axes[row, col].axis('off')
                else:  # Reconstructed image row
                    img_idx = ((row - 1) // 2) * cols + col
                    if img_idx >= batch_size:
                        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot with {batch_size} images saved to {output_path}")

def calculate_metrics(original_images, reconstructed_images):
    """Calculate reconstruction metrics"""
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(reconstructed_images, original_images)
        
        # PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # SSIM (simplified version)
        def ssim(img1, img2):
            mu1 = F.avg_pool2d(img1, 3, 1, 1)
            mu2 = F.avg_pool2d(img2, 3, 1, 1)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
        
        ssim_val = ssim(original_images, reconstructed_images)
        
        return {
            'MSE': mse.item(),
            'PSNR': psnr.item(),
            'SSIM': ssim_val.item()
        }

def run_attack_on_model(model_path, output_dir="gradient_attack_results", num_iterations=10000, learning_rate=0.01, batch_size=32):
    """Run gradient inversion attack on a specific model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Attack parameters: iterations={num_iterations}, lr={learning_rate}, batch_size={batch_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    model = load_trained_model(model_path, device)
    
    # Get real data batch
    print("Loading real CIFAR-10 data...")
    real_images, real_labels = get_real_data_batch(batch_size=batch_size)
    real_images = real_images.to(device)
    real_labels = real_labels.to(device)
    
    print(f"Real data batch shape: {real_images.shape}")
    print(f"Real labels: {real_labels}")
    
    # Compute target gradients from real data
    print("Computing target gradients...")
    loss_fn = nn.CrossEntropyLoss()
    target_gradients, target_loss = compute_gradients(model, real_images, real_labels, loss_fn)
    
    print(f"Target loss: {target_loss.item():.4f}")
    print(f"Number of gradient tensors: {len(target_gradients)}")
    
    # Run gradient inversion attack
    reconstructed_images, history = gradient_inversion_attack(
        model, target_gradients, real_images.shape[0], num_iterations=num_iterations, lr=learning_rate
    )
    
    # Calculate metrics
    metrics = calculate_metrics(real_images, reconstructed_images)
    
    # Save results
    model_name = Path(model_path).stem
    comparison_path = os.path.join(output_dir, f"{model_name}_comparison.png")
    plot_comparison(real_images, reconstructed_images, comparison_path)
    
    # Save metrics
    results_path = os.path.join(output_dir, f"{model_name}_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Gradient Inversion Attack Results\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Attack Configuration:\n")
        f.write(f"  - Max iterations: {num_iterations}\n")
        f.write(f"  - Learning rate: {learning_rate}\n")
        f.write(f"  - Batch size: {batch_size}\n")
        f.write(f"  - Target loss: {target_loss.item():.6f}\n\n")
        f.write("Reconstruction Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
    
    # Print results
    print(f"\n{'='*50}")
    print("GRADIENT INVERSION ATTACK RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Target Loss: {target_loss.item():.6f}")
    print(f"Batch Size: {real_images.shape[0]}")
    print("\nReconstruction Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Comparison plot: {comparison_path}")
    print(f"  - Results file: {results_path}")
    
    return metrics

def test_multiple_models():
    """Test gradient inversion attacks on multiple trained models"""
    models_dir = Path("../results/CFL/client_models/round_models")
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    # Test different rounds
    test_models = [
        "c0_1_round_1_model.npz",   # Early training
        "c0_2_round_1_model.npz",   # Mid training
        "c0_3_round_1_model.npz"   # Final model
    ]
    
    results_summary = {}
    
    for model_name in test_models:
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"\n{'='*60}")
            print(f"Testing gradient attack on: {model_name}")
            print(f"{'='*60}")
            
            output_dir = f"attack_results_{model_name.replace('.npz', '')}"
            
            try:
                metrics = run_attack_on_model(str(model_path), output_dir)
                results_summary[model_name] = metrics
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Model not found: {model_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("ATTACK SUMMARY")
    print(f"{'='*60}")
    
    for model_name, metrics in results_summary.items():
        print(f"\n{model_name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

def test_with_more_iterations():
    """Test with higher number of iterations for better quality"""
    models_dir = Path("../results/CFL/global_models")
    
    # Test with more iterations on the final model
    model_path = models_dir / "node_n1_round_10_global_model.npz"
    
    if model_path.exists():
        print("Testing with MORE ITERATIONS for better reconstruction quality...")
        print("This will take longer but should produce better results.")
        
        # Test with different iteration counts
        iteration_tests = [
            (2000, "2k_iterations"),
            (5000, "5k_iterations"),
            (8000, "8k_iterations")
        ]
        
        for iterations, suffix in iteration_tests:
            print(f"\n{'='*60}")
            print(f"Testing with {iterations} iterations")
            print(f"{'='*60}")
            
            output_dir = f"high_quality_attack_{suffix}"
            
            try:
                metrics = run_attack_on_model(
                    str(model_path), 
                    output_dir, 
                    num_iterations=iterations,
                    learning_rate=0.01,
                    batch_size=4
                )
                print(f"Results for {iterations} iterations: {metrics}")
            except Exception as e:
                print(f"Error with {iterations} iterations: {e}")
    else:
        print(f"Model not found: {model_path}")

def test_client_models():
    """Test gradient inversion attacks on individual client models (should be much more vulnerable)"""
    # First check for individual client models (new format)
    client_models_dir = Path("../results/CFL/client_models/round_models")
    cluster_models_dir = Path("../results/CFL/cluster_models")
    
    test_models = []
    
    # Check for individual client round models (new format)
    if client_models_dir.exists():
        print(f"Found individual client models directory: {client_models_dir}")
        # Test specific individual client models from different rounds
        individual_models = [
            "c0_1_round_1_model.npz",   # Early training, client 1
            "c0_1_round_5_model.npz",   # Mid training, client 1  
            "c0_1_round_10_model.npz",  # Final round, client 1
            "c0_2_round_1_model.npz",   # Early training, client 2
            "c0_2_round_10_model.npz",  # Final round, client 2 (for comparison)
        ]
        
        for model_name in individual_models:
            model_path = client_models_dir / model_name
            if model_path.exists():
                test_models.append((str(model_path), f"individual_{model_name}"))
    
    # Also check for cluster models (existing format)
    if cluster_models_dir.exists():
        print(f"Found cluster models directory: {cluster_models_dir}")
        cluster_models = [
            "c0_1_round_1_cluster_sum.npz",   # Early training, client 1
            "c0_1_round_5_cluster_sum.npz",   # Mid training, client 1  
            "c0_1_round_10_cluster_sum.npz",  # Final round, client 1
            "c0_2_round_10_cluster_sum.npz",  # Final round, client 2 (for comparison)
        ]
        
        for model_name in cluster_models:
            model_path = cluster_models_dir / model_name
            if model_path.exists():
                test_models.append((str(model_path), f"cluster_{model_name}"))
    
    if not test_models:
        print("No client models found in either individual or cluster directories")
        return
    
    print("=" * 80)
    print("TESTING GRADIENT ATTACKS ON CLIENT MODELS")
    print("Individual client models should be MUCH more vulnerable than global models!")
    print("=" * 80)
    
    results_summary = {}
    
    for model_path, model_description in test_models:
        print(f"\n{'='*60}")
        print(f"Testing gradient attack on: {model_description}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")
        
        output_dir = f"client_attack_{model_description.replace('.npz', '').replace('/', '_')}"
        
        try:
            # Use more aggressive parameters for client models
            metrics = run_attack_on_model(
                model_path, 
                output_dir,
                num_iterations=3000,   # More iterations for better quality
                learning_rate=0.01,    # Standard learning rate
                batch_size=2           # Smaller batch for better reconstruction
            )
            results_summary[model_description] = metrics
            
            # Print immediate results
            print(f"CLIENT MODEL ATTACK RESULTS:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
                
        except Exception as e:
            print(f"Error testing {model_description}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("CLIENT MODEL ATTACK SUMMARY")
    print(f"{'='*80}")
    
    for model_description, metrics in results_summary.items():
        print(f"\n{model_description}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
            
    return results_summary

def test_best_attack_scenarios():
    """Test the most promising attack scenarios"""
    print("=" * 80)
    print("TESTING BEST ATTACK SCENARIOS")
    print("=" * 80)
    
    # Test early training client model (should be most vulnerable)
    early_client_path = Path("../results/CFL/cluster_models/c0_1_round_1_cluster_sum.npz")
    
    if early_client_path.exists():
        print("\n🎯 TESTING EARLY TRAINING CLIENT MODEL (Most Vulnerable)")
        print("This should give the best reconstruction results!")
        
        scenarios = [
            (1000, 0.02, 1, "quick_test"),
            (3000, 0.01, 2, "standard_quality"),
            (6000, 0.005, 1, "high_quality"),
            (10000, 0.003, 1, "maximum_quality")
        ]
        
        best_metrics = None
        best_scenario = None
        
        for iterations, lr, batch_size, scenario_name in scenarios:
            print(f"\n📊 Scenario: {scenario_name}")
            print(f"   Iterations: {iterations}, LR: {lr}, Batch: {batch_size}")
            
            try:
                metrics = run_attack_on_model(
                    str(early_client_path),
                    f"best_attack_{scenario_name}",
                    num_iterations=iterations,
                    learning_rate=lr,
                    batch_size=batch_size
                )
                
                # Track best results (higher PSNR is better)
                if best_metrics is None or metrics['PSNR'] > best_metrics['PSNR']:
                    best_metrics = metrics
                    best_scenario = scenario_name
                    
                print(f"✅ {scenario_name}: PSNR={metrics['PSNR']:.3f}, MSE={metrics['MSE']:.6f}")
                
            except Exception as e:
                print(f"❌ {scenario_name} failed: {e}")
        
        if best_metrics:
            print(f"\n🏆 BEST RECONSTRUCTION: {best_scenario}")
            print(f"   PSNR: {best_metrics['PSNR']:.6f}")
            print(f"   MSE: {best_metrics['MSE']:.6f}")
            print(f"   SSIM: {best_metrics['SSIM']:.6f}")
    else:
        print("❌ Early client model not found!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--high-quality":
            test_with_more_iterations()
        elif sys.argv[1] == "--client-models":
            test_client_models()
        elif sys.argv[1] == "--best-attack":
            test_best_attack_scenarios()
        else:
            print("Usage: python3 simple_gradient_attack.py [--high-quality|--client-models|--best-attack]")
    else:
        test_multiple_models() 