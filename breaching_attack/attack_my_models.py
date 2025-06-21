"""
Breaching attack on trained federated learning models
This script tests gradient inversion attacks on models trained with the cluster-shuffling FL system.
"""

import breaching
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

class SimpleNet(nn.Module):
    """SimpleNet architecture matching the trained models"""
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding='valid')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding='valid')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_trained_model(model_path, device):
    """Load a trained model from .npz file"""
    logger.info(f"Loading model from {model_path}")
    
    # Load the .npz file
    model_data = np.load(model_path)
    
    # Create SimpleNet model
    model = SimpleNet(num_classes=10)
    
    # Convert numpy arrays to torch tensors and load into model
    state_dict = {}
    
    # Map the parameter names from the saved model to PyTorch names
    param_mapping = {
        'param_0': 'conv1.weight',
        'param_1': 'conv1.bias', 
        'param_2': 'conv2.weight',
        'param_3': 'conv2.bias',
        'param_4': 'fc1.weight',
        'param_5': 'fc1.bias',
        'param_6': 'fc2.weight', 
        'param_7': 'fc2.bias',
        'param_8': 'fc3.weight',
        'param_9': 'fc3.bias'
    }
    
    for saved_name, pytorch_name in param_mapping.items():
        if saved_name in model_data:
            state_dict[pytorch_name] = torch.from_numpy(model_data[saved_name])
            logger.info(f"Loaded {pytorch_name}: {model_data[saved_name].shape}")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully!")
    return model

def run_breaching_attack(model_path, output_dir="breaching_results", num_iterations=8000):
    """Run breaching attack on a trained model"""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    trained_model = load_trained_model(model_path, device)
    
    # Setup breaching configuration for CIFAR-10
    cfg = breaching.get_config(overrides=[
        "case=4_fedavg_small_scale",
        "case/data=CIFAR10"
    ])
    
    # Configure the attack
    cfg.attack.optim.max_iterations = num_iterations
    cfg.case.user.num_data_points = 8  # Number of data points to reconstruct
    cfg.case.user.num_local_updates = 1
    cfg.case.user.num_data_per_local_update_step = 8
    cfg.case.user.provide_labels = True  # Provide labels to make attack easier
    
    # Setup torch configuration
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    
    # Construct the case (this creates a template)
    user, server, template_model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    
    # Replace the server model with our trained model
    server.model = trained_model
    server.model.name = "SimpleNet"
    
    # Prepare the attacker
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    
    # Print overview
    breaching.utils.overview(server, user, attacker)
    
    # Simulate a federated learning round
    logger.info("Simulating federated learning round...")
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)
    
    # Plot and save original data
    plt.figure(figsize=(12, 6))
    user.plot(true_user_data)
    plt.title('Original User Data (Ground Truth)')
    original_path = os.path.join(output_dir, 'original_data.png')
    plt.savefig(original_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved original data plot to {original_path}")
    
    # Perform the reconstruction attack
    logger.info("Starting gradient inversion attack...")
    reconstructed_user_data, stats = attacker.reconstruct(
        [server_payload], [shared_data], {}, dryrun=cfg.dryrun
    )
    
    # Calculate reconstruction metrics
    logger.info("Calculating reconstruction metrics...")
    metrics = breaching.analysis.report(
        reconstructed_user_data, true_user_data, [server_payload],
        server.model, order_batch=True, compute_full_iip=False,
        cfg_case=cfg.case, setup=setup
    )
    
    # Plot and save reconstructed data
    plt.figure(figsize=(12, 6))
    user.plot(reconstructed_user_data)
    plt.title('Reconstructed User Data (Attack Result)')
    reconstructed_path = os.path.join(output_dir, 'reconstructed_data.png')
    plt.savefig(reconstructed_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved reconstructed data plot to {reconstructed_path}")
    
    # Save results to file
    results_path = os.path.join(output_dir, 'attack_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Breaching Attack Results for {model_path}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Attack Configuration:\n")
        f.write(f"  - Max iterations: {num_iterations}\n")
        f.write(f"  - Data points: {cfg.case.user.num_data_points}\n")
        f.write(f"  - Local updates: {cfg.case.user.num_local_updates}\n")
        f.write(f"  - Labels provided: {cfg.case.user.provide_labels}\n\n")
        
        f.write("Reconstruction Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  - {key}: {value}\n")
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("BREACHING ATTACK RESULTS")
    logger.info("="*50)
    logger.info(f"Model: {model_path}")
    logger.info(f"Attack iterations: {num_iterations}")
    logger.info(f"Data points reconstructed: {cfg.case.user.num_data_points}")
    logger.info("\nReconstruction Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - Original data: {original_path}")
    logger.info(f"  - Reconstructed data: {reconstructed_path}")
    logger.info(f"  - Full results: {results_path}")
    
    return metrics

def test_multiple_models():
    """Test breaching attacks on multiple trained models"""
    
    # Find all available models
    models_dir = Path("../results/CFL/global_models")
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return
    
    model_files = list(models_dir.glob("*.npz"))
    logger.info(f"Found {len(model_files)} trained models")
    
    # Test a few different models
    test_models = [
        "node_n1_round_1_global_model.npz",  # Early training
        "node_n1_round_5_global_model.npz",  # Mid training  
        "node_n1_round_10_global_model.npz"  # Final model
    ]
    
    results_summary = {}
    
    for model_name in test_models:
        model_path = models_dir / model_name
        if model_path.exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing breaching attack on: {model_name}")
            logger.info(f"{'='*60}")
            
            output_dir = f"breaching_results_{model_name.replace('.npz', '')}"
            
            try:
                metrics = run_breaching_attack(str(model_path), output_dir)
                results_summary[model_name] = metrics
            except Exception as e:
                logger.error(f"Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"Model not found: {model_path}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("ATTACK SUMMARY")
    logger.info(f"{'='*60}")
    
    for model_name, metrics in results_summary.items():
        logger.info(f"\n{model_name}:")
        # Print key metrics
        for key in ['MSE', 'PSNR', 'LPIPS', 'SSIM']:
            if key in metrics:
                logger.info(f"  {key}: {metrics[key]:.6f}")

if __name__ == "__main__":
    # Test multiple models
    test_multiple_models() 