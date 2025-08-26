"""
Gradient Inversion Attack implementation for federated learning privacy evaluation
"""

import torch
import numpy as np
import os
import time
import sys
import inversefed
from models import Net
from config import settings

from .utils import (
    load_gradient_data, load_model_data, find_latest_experiment, 
    find_available_clients, calculate_attack_metrics, save_attack_results,
    generate_attack_summary_table, generate_privacy_evaluation_report
)
from .attack_configs import ATTACK_CONFIGS


class GradientInversionAttacker:
    """
    Modular gradient inversion attacker for evaluating federated learning privacy
    """
    
    def __init__(self, attack_config='default', device='auto', output_dir='attack_results'):
        """
        Initialize gradient inversion attacker
        
        Args:
            attack_config: Either a config name from ATTACK_CONFIGS or a custom config dict
            device: Device to run on ('auto', 'cuda', 'cpu')
            output_dir: Directory to save attack results
        """
        self.device = self._setup_device(device)
        self.output_dir = output_dir
        
        # Set attack configuration
        if isinstance(attack_config, str):
            if attack_config not in ATTACK_CONFIGS:
                raise ValueError(f"Unknown attack config: {attack_config}. Available: {list(ATTACK_CONFIGS.keys())}")
            self.attack_config = ATTACK_CONFIGS[attack_config].copy()
            self.config_name = attack_config
        else:
            self.attack_config = attack_config.copy()
            self.config_name = 'custom'
        
        # Setup inversefed
        self.setup = inversefed.utils.system_startup()
        self.defs = inversefed.training_strategy('conservative')
        
        # Load CIFAR-10 dataset for ground truth reconstruction
        self.loss_fn, _, self.validloader = inversefed.construct_dataloaders(
            'CIFAR10', self.defs, data_path='./datasets/cifar10'
        )
        
        # Normalization constants
        self.dm = torch.as_tensor(inversefed.consts.cifar10_mean, **self.setup)[:, None, None]
        self.ds = torch.as_tensor(inversefed.consts.cifar10_std, **self.setup)[:, None, None]
        
        print(f"✅ GradientInversionAttacker initialized")
        print(f"   Config: {self.config_name}")
        print(f"   Device: {self.device}")
        print(f"   Output: {self.output_dir}")
        
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def attack_single_client(self, exp_dir, round_num, client_id, save_individual=True):
        """
        Attack a specific client's gradients
        
        Args:
            exp_dir: Experiment directory path
            round_num: Training round number
            client_id: Client identifier
            save_individual: Save individual image reconstructions
            
        Returns:
            dict: Attack results and metrics
        """
        print(f"\n{'='*70}")
        print(f"=== Attacking Round {round_num}, Client {client_id} ==={' ':<20}")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        try:
            # Load data
            grad_data = load_gradient_data(exp_dir, round_num, client_id)
            model_data = load_model_data(exp_dir, round_num, client_id)
            
            # Extract data from ModelManager format
            saved_gradients = grad_data['gradients']
            batch_images = grad_data['batch_images']
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
            
            # Create and setup model
            model = Net(
                num_classes=len(settings.get('classes', range(10))), 
                arch=settings['arch'], 
                pretrained=settings['pretrained'], 
                input_size=(32, 32)
            )
            model.load_state_dict(model_state)
            model.to(**self.setup)
            model.eval()
            
            # Prepare data
            ground_truth = batch_images.to(**self.setup)
            true_labels = batch_labels.to(device=self.setup['device'])
            gradients_for_attack = [g.to(**self.setup) for g in saved_gradients]
            
            # Gradient statistics
            gradient_norm = torch.stack([g.norm() for g in gradients_for_attack]).mean().item()
            print(f"\nGradient Statistics:")
            print(f"  Gradient norm: {gradient_norm:.4f}")
            print(f"  Number of gradient tensors: {len(gradients_for_attack)}")
            
            # Verify gradients
            fresh_gradient_norm, fresh_loss = self._verify_gradients(
                model, ground_truth, true_labels, gradient_norm
            )
            
            # Perform attack
            print(f"\nStarting reconstruction with {self.attack_config['restarts']} restarts...")
            print(f"Reconstructing {batch_images.shape[0]} images of shape (3, 32, 32)")
            sys.stdout.flush()
            
            start_time = time.time()
            output, stats = self._perform_reconstruction(
                model, gradients_for_attack, true_labels, batch_images.shape[0]
            )
            attack_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_attack_metrics(output, ground_truth, (self.dm, self.ds))
            
            # Print results
            self._print_attack_results(round_num, client_id, attack_time, stats, metrics)
            
            # Save images
            save_attack_results(
                output, ground_truth, (self.dm, self.ds), 
                round_num, client_id, self.output_dir, save_individual
            )
            
            # Return comprehensive results
            return {
                'round': round_num,
                'client_id': client_id,
                'psnr': metrics['psnr_avg'],
                'mse': metrics['mse'],
                'attack_time': attack_time,
                'gradient_norm': gradient_norm,
                'fresh_gradient_norm': fresh_gradient_norm,
                'loss': loss,
                'fresh_loss': fresh_loss,
                'reconstruction_loss': stats['opt'],
                'per_image_psnr': metrics['per_image_psnr'],
                'accuracy': accuracy,
                'metrics': metrics,
                'config_name': self.config_name
            }
            
        except Exception as e:
            print(f"Error attacking round {round_num}, client {client_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _verify_gradients(self, model, ground_truth, true_labels, saved_gradient_norm):
        """Verify saved gradients by computing fresh ones"""
        print("Computing fresh gradients for verification...")
        model.zero_grad()
        target_loss_output = self.loss_fn(model(ground_truth), true_labels)
        
        # Handle tuple output from loss function
        if isinstance(target_loss_output, (tuple, list)):
            target_loss = target_loss_output[0]
        else:
            target_loss = target_loss_output
            
        fresh_gradients = torch.autograd.grad(target_loss, model.parameters())
        fresh_gradient_norm = torch.stack([g.norm() for g in fresh_gradients]).mean().item()
        
        print(f"  Fresh gradient norm: {fresh_gradient_norm:.4f}")
        print(f"  Fresh loss: {target_loss.item():.4f}")
        print(f"  Gradient norm ratio: {saved_gradient_norm/fresh_gradient_norm:.4f}")
        
        return fresh_gradient_norm, target_loss.item()
    
    def _perform_reconstruction(self, model, gradients, labels, num_images):
        """Perform the actual gradient inversion reconstruction"""
        rec_machine = inversefed.GradientReconstructor(
            model, (self.dm, self.ds), self.attack_config, num_images=num_images
        )
        output, stats = rec_machine.reconstruct(gradients, labels, img_shape=(3, 32, 32))
        return output, stats
    
    def _print_attack_results(self, round_num, client_id, attack_time, stats, metrics):
        """Print formatted attack results"""
        print(f"\n=== Results for Round {round_num}, Client {client_id} ===")
        print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
        print(f"Reconstruction loss: {stats['opt']:.4f}")
        print(f"Average PSNR: {metrics['psnr_avg']:.2f} dB")
        print(f"Best image PSNR: {metrics['psnr_best']:.2f} dB")
        print(f"Worst image PSNR: {metrics['psnr_worst']:.2f} dB")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"Success rate (>20 dB): {metrics['success_rate_20db']:.1f}%")
        print(f"Success rate (>25 dB): {metrics['success_rate_25db']:.1f}%")
        sys.stdout.flush()
    
    def attack_experiment(self, exp_dir=None, attack_rounds=None, clients=None):
        """
        Attack an entire experiment (multiple clients and rounds)
        
        Args:
            exp_dir: Experiment directory (if None, finds latest)
            attack_rounds: List of rounds to attack (if None, uses config)
            clients: List of clients to attack (if None, finds all available)
            
        Returns:
            list: List of attack results
        """
        print(f"=== Gradient Inversion Attack ({self.config_name}) ===")
        print(f"Device: {self.device}")
        print(f"Architecture: {settings['arch']}")
        sys.stdout.flush()
        
        # Find experiment directory
        if exp_dir is None:
            try:
                exp_dir = find_latest_experiment()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return []
        
        # Determine attack rounds
        if attack_rounds is None:
            attack_rounds = settings.get('save_gradients_rounds', [1, 2, 3])
        
        print(f"Attack rounds: {attack_rounds}")
        
        # Find available clients
        if clients is None:
            clients = find_available_clients(exp_dir, attack_rounds)
        
        print(f"Found clients with saved gradients: {clients}")
        
        results = []
        
        # Attack each client in each round
        for round_num in attack_rounds:
            for client_id in clients:
                result = self.attack_single_client(exp_dir, round_num, client_id)
                if result:
                    results.append(result)
                    
                    # Save intermediate results
                    self._save_intermediate_results(results, exp_dir)
        
        # Generate final report
        self._generate_final_report(results)
        
        return results
    
    def _save_intermediate_results(self, results, exp_dir):
        """Save intermediate attack results"""
        torch.save({
            'results': results,
            'attack_config': self.attack_config,
            'config_name': self.config_name,
            'architecture': settings['arch'],
            'method': 'gradient_inversion',
            'experiment_dir': exp_dir
        }, f'{self.output_dir}/attack_results.pt')
    
    def _generate_final_report(self, results):
        """Generate comprehensive final attack report"""
        if not results:
            print("No successful attacks to report")
            return
        
        # Summary table
        generate_attack_summary_table(results)
        
        # Privacy evaluation
        generate_privacy_evaluation_report(results, f"Baseline ({self.config_name} attack)")
        
        print(f"\n✅ Gradient inversion attack complete!")
        print(f"Check '{self.output_dir}/' for detailed results")
        sys.stdout.flush()
    
    def compare_configurations(self, config_names, exp_dir=None):
        """
        Compare multiple attack configurations on the same experiment
        
        Args:
            config_names: List of configuration names to compare
            exp_dir: Experiment directory
            
        Returns:
            dict: Comparison results
        """
        print(f"=== Comparing Attack Configurations ===")
        print(f"Configurations: {config_names}")
        
        comparison_results = {}
        
        for config_name in config_names:
            print(f"\n--- Testing {config_name} configuration ---")
            
            # Create temporary attacker with this config
            temp_attacker = GradientInversionAttacker(
                attack_config=config_name,
                device=self.device,
                output_dir=f"{self.output_dir}_{config_name}"
            )
            
            # Attack with this configuration
            results = temp_attacker.attack_experiment(exp_dir)
            comparison_results[config_name] = results
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def _generate_comparison_report(self, comparison_results):
        """Generate a comparison report across different configurations"""
        print(f"\n{'='*80}")
        print("=== ATTACK CONFIGURATION COMPARISON ===")
        print(f"{'='*80}")
        
        print(f"{'Config':<15} {'Avg PSNR':<10} {'Success Rate':<12} {'Avg Time (min)':<15}")
        print(f"{'-'*60}")
        
        for config_name, results in comparison_results.items():
            if results:
                avg_psnr = np.mean([r['psnr'] for r in results])
                success_rate = len([r for r in results if r['psnr'] > 20]) / len(results) * 100
                avg_time = np.mean([r['attack_time'] for r in results]) / 60
                
                print(f"{config_name:<15} {avg_psnr:<10.2f} {success_rate:<12.1f}% {avg_time:<15.1f}")
            else:
                print(f"{config_name:<15} {'No results':<40}")
        
        print(f"\n🎯 Use this comparison to select the optimal attack configuration for your research!")