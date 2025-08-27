#!/usr/bin/env python3
"""
GIFD Attack Runner - Generative Gradient Inversion with Feature Domain Optimization

This script runs the GIFD attack method on saved gradients from federated learning experiments.
GIFD uses pre-trained generative models (StyleGAN2, BigGAN) as strong priors for high-quality
reconstruction.

Examples:
    # Run GIFD with StyleGAN2 on latest experiment
    python run_gifd.py --gan stylegan2
    
    # Run GIFD with BigGAN on specific experiment
    python run_gifd.py --experiment cifar10_classic_c3_r3 --gan biggan
    
    # Attack specific rounds and clients
    python run_gifd.py --rounds 1 2 --clients c0_1 c0_2
    
    # Use aggressive GIFD configuration
    python run_gifd.py --config aggressive --gan stylegan2
"""

import os
import sys
import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
import pickle
import time
from datetime import datetime

# Add GIFD path to system
GIFD_PATH = os.path.join(os.path.dirname(__file__), 'GIFD_Gradient_Inversion_Attack')
sys.path.insert(0, GIFD_PATH)

# Import GIFD components
import inversefed
from inversefed import GradientReconstructor
from inversefed.utils import save_to_table

# Import our components
from attacks.utils import (
    load_gradient_data, load_model_data, find_latest_experiment,
    calculate_attack_metrics, save_attack_results
)
from models import Net
from config import settings
import torchvision.transforms as transforms


# GIFD Attack Configurations
GIFD_CONFIGS = {
    'quick_test': {
        'restarts': 1,
        'max_iterations': 1000,
        'gias_iterations': 2000,
        'steps': [500, 500],  # For multi-stage optimization
        'lr_io': [0.1, 0.1],
        'total_variation': 1e-4,
        'image_norm': 1e-6,
    },
    'default': {
        'restarts': 2,
        'max_iterations': 2000,
        'gias_iterations': 4000,
        'steps': [1000, 1000, 1000, 1000],
        'lr_io': [0.1, 0.1, 0.05, 0.05],
        'total_variation': 1e-4,
        'image_norm': 1e-6,
    },
    'aggressive': {
        'restarts': 4,
        'max_iterations': 4000,
        'gias_iterations': 8000,
        'steps': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        'lr_io': [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
        'total_variation': 1e-4,
        'image_norm': 1e-6,
    }
}


class GIFDAttacker:
    """GIFD Attack wrapper for federated learning gradient inversion"""
    
    def __init__(self, gan_type='stylegan2', config='default', device='auto', use_ada=False):
        """
        Initialize GIFD attacker
        
        Args:
            gan_type: Type of GAN to use ('stylegan2', 'stylegan2-ada', 'biggan', 'none')
            config: Attack configuration ('quick_test', 'default', 'aggressive')
            device: Device to run on ('auto', 'cuda', 'cpu')
            use_ada: Use StyleGAN2-ADA with pkl files
        """
        self.gan_type = gan_type
        self.use_ada = use_ada or (gan_type == 'stylegan2-ada')
        self.device = self._setup_device(device)
        
        # Get configuration
        if config not in GIFD_CONFIGS:
            raise ValueError(f"Unknown config: {config}. Available: {list(GIFD_CONFIGS.keys())}")
        self.config = GIFD_CONFIGS[config].copy()
        self.config_name = config
        
        # Setup GIFD components
        self.setup = inversefed.utils.system_startup()
        self.defs = inversefed.training_strategy('conservative')
        
        # Dataset configuration
        self.dataset_name = 'CIFAR10'
        self.num_classes = 10
        self.image_shape = (3, 32, 32)
        
        # Load dataset for ground truth comparison
        self.loss_fn, _, self.validloader = inversefed.construct_dataloaders(
            self.dataset_name, self.defs, data_path='./datasets/cifar10'
        )
        
        # Normalization constants
        self.dm = torch.as_tensor(inversefed.consts.cifar10_mean, **self.setup)[:, None, None]
        self.ds = torch.as_tensor(inversefed.consts.cifar10_std, **self.setup)[:, None, None]
        
        print(f"✅ GIFD Attacker initialized")
        print(f"   GAN Type: {gan_type}")
        print(f"   Config: {self.config_name}")
        print(f"   Device: {self.device}")
        
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _prepare_gifd_config(self):
        """Prepare full GIFD configuration"""
        gifd_config = {
            # Base settings
            'cost_fn': 'sim_cmpr0',
            'indices': 'def',
            'weights': 'equal',
            'init': 'randn',
            'optim': 'adam' if self.gan_type == 'none' else 'GAN_based',
            'lr': 0.1,
            'lr_decay': True,
            'dataset': self.dataset_name,
            
            # From our config
            'restarts': self.config['restarts'],
            'max_iterations': self.config['max_iterations'],
            'total_variation': self.config['total_variation'],
            'image_norm': self.config['image_norm'],
            'bn_stat': 0,
            'z_norm': 0,
            'group_lazy': 0,
            
            # GAN settings
            'generative_model': '',
            'gen_dataset': '',
            'gifd': False,
            'gias': False,
            'gias_lr': 0.01,
            'gias_iterations': 0,
            
            # Multi-stage optimization
            'steps': [],
            'lr_io': [],
            'start_layer': 0,
            'end_layer': 8,
            
            # Projection settings
            'do_project_gen_out': False,
            'do_project_noises': False, 
            'do_project_latent': False,
            'max_radius_gen_out': [],
            'max_radius_noises': [],
            'max_radius_latent': [],
            
            # Other
            'ckpt': [],
            'project': False,
            'defense_method': [],
            'defense_setting': [],
            'num_sample': 10,
            'KLD': 0,
            'cma_budget': 0,
            'lr_same_pace': False,
            'ggl': False,
            'yin': False,
            'geiping': False,
        }
        
        # Configure based on GAN type
        if self.gan_type == 'dcgan':
            gifd_config.update({
                'generative_model': 'cifar10_dcgan',
                'gen_dataset': 'CIFAR10',
                'gifd': True,
                'gias': True,
                'gias_iterations': self.config['gias_iterations'],
                'steps': self.config['steps'],
                'lr_io': self.config['lr_io'],
            })
            
        elif self.gan_type == 'stylegan2' or self.gan_type == 'stylegan2-ada':
            if self.use_ada:
                # Use StyleGAN2-ADA with pkl files
                gifd_config.update({
                    'generative_model': 'stylegan2_ada',
                    'gen_dataset': 'C10',  # CIFAR-10 for ADA
                    'gifd': True,
                    'gias': True,
                    'gias_iterations': self.config['gias_iterations'],
                    'steps': self.config['steps'],
                    'lr_io': self.config['lr_io'],
                })
            else:
                gifd_config.update({
                    'generative_model': 'stylegan2_io',
                    'gen_dataset': 'CIFAR10',  # Will use CIFAR-10 features
                    'gifd': True,
                    'gias': True,
                    'gias_iterations': self.config['gias_iterations'],
                    'steps': self.config['steps'],
                    'lr_io': self.config['lr_io'],
                    'project': True,
                    'do_project_gen_out': True,
                    'do_project_noises': True,
                    'do_project_latent': True,
                    'max_radius_gen_out': [1000, 2000, 3000, 4000] * 2,
                    'max_radius_noises': [1000, 2000, 3000, 4000] * 2,
                    'max_radius_latent': [1000, 2000, 3000, 4000] * 2,
                })
            
        elif self.gan_type == 'biggan':
            gifd_config.update({
                'generative_model': 'BigGAN',
                'gen_dataset': 'ImageNet',
                'gifd': True,
                'gias': True,
                'gias_iterations': self.config['gias_iterations'],
                'steps': self.config['steps'],
                'lr_io': self.config['lr_io'],
            })
        
        elif self.gan_type == 'none':
            # GAN-free method (standard optimization)
            gifd_config.update({
                'geiping': True,  # Use Geiping et al. method
                'cost_fn': 'sim_cmpr0',
                'image_norm': self.config['image_norm'],
                'group_lazy': 1e-2,
            })
        
        return gifd_config
    
    def attack_client(self, exp_dir, round_num, client_id):
        """
        Run GIFD attack on a single client's gradients
        
        Args:
            exp_dir: Experiment directory path
            round_num: Training round number
            client_id: Client identifier
            
        Returns:
            dict: Attack results with reconstructed images and metrics
        """
        print(f"\n{'='*70}")
        print(f"🎯 GIFD Attack - Round {round_num}, Client {client_id}")
        print(f"{'='*70}")
        
        try:
            # Load gradient and model data
            grad_data = load_gradient_data(exp_dir, round_num, client_id)
            model_data = load_model_data(exp_dir, round_num, client_id)
            
            # Extract data
            saved_gradients = grad_data['gradients']
            batch_images = grad_data['batch_images'] 
            batch_labels = grad_data['batch_labels']
            model_state = model_data.get('model_state_dict', model_data.get('model_state'))
            
            print(f"📊 Loaded gradient data:")
            print(f"   Batch size: {len(batch_labels)}")
            print(f"   Labels: {batch_labels.tolist()}")
            
            # Create model and load state
            model = Net(
                arch=grad_data.get('model_architecture', 'convnet'),
                num_classes=self.num_classes
            )
            model.load_state_dict(model_state)
            model = model.to(self.device)
            model.eval()
            
            # Convert gradients to GIFD format
            input_gradient = []
            for grad in saved_gradients:
                if isinstance(grad, torch.Tensor):
                    input_gradient.append(grad.to(self.device))
                else:
                    input_gradient.append(torch.tensor(grad, device=self.device))
            
            # Prepare labels
            if isinstance(batch_labels, torch.Tensor):
                labels = batch_labels.clone().detach().to(self.device)
            else:
                labels = torch.tensor(batch_labels, device=self.device, dtype=torch.long)
            
            # Initialize GIFD reconstructor with config
            gifd_config = self._prepare_gifd_config()
            rec_machine = GradientReconstructor(
                model, self.device, 
                mean_std=(self.dm, self.ds), 
                config=gifd_config, 
                num_images=len(batch_labels), 
                G=None  # Let GradientReconstructor load the GAN based on config
            )
            
            # Run reconstruction
            print(f"\n🔬 Starting GIFD reconstruction...")
            print(f"   Method: {self.gan_type}")
            print(f"   Restarts: {gifd_config['restarts']}")
            print(f"   Iterations: {gifd_config['max_iterations']}")
            
            start_time = time.time()
            
            # Perform reconstruction
            output, stats = rec_machine.reconstruct(
                input_gradient, labels, 
                img_shape=self.image_shape,
                dryrun=False
            )
            
            reconstruction_time = time.time() - start_time
            
            # Process results
            if len(output) > 0:
                # Extract best reconstruction
                if isinstance(output[0], list):
                    # Multiple methods tested, take best
                    method_name, rec_images, method_stats = output[0]
                    print(f"   Method used: {method_name}")
                else:
                    rec_images = output
                    method_stats = stats
                
                # Ensure proper format
                if not isinstance(rec_images, torch.Tensor):
                    rec_images = rec_images[0] if isinstance(rec_images, list) else rec_images
                
                # Denormalize for metric calculation
                rec_images_denorm = rec_images * self.ds + self.dm
                
                # Convert ground truth to proper format
                ground_truth = torch.tensor(batch_images, device=self.device)
                if ground_truth.shape[1:] != (3, 32, 32):
                    ground_truth = ground_truth.permute(0, 3, 1, 2) if ground_truth.shape[-1] == 3 else ground_truth
                ground_truth = ground_truth.float() / 255.0 if ground_truth.max() > 1 else ground_truth
                
                # Calculate metrics
                metrics = calculate_attack_metrics(rec_images_denorm, ground_truth)
                
                # Prepare results
                results = {
                    'round': round_num,
                    'client_id': client_id,
                    'reconstructed_images': rec_images_denorm.cpu(),
                    'ground_truth': ground_truth.cpu(),
                    'labels': batch_labels,
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'mse': metrics['mse'],
                    'lpips': metrics.get('lpips', 0),
                    'reconstruction_time': reconstruction_time,
                    'method': self.gan_type,
                    'config': self.config_name,
                    'stats': method_stats if 'method_stats' in locals() else {}
                }
                
                print(f"\n✅ Reconstruction complete!")
                print(f"   PSNR: {metrics['psnr']:.2f} dB")
                print(f"   SSIM: {metrics['ssim']:.4f}")
                print(f"   MSE: {metrics['mse']:.4f}")
                print(f"   Time: {reconstruction_time:.2f}s")
                
                # Save results
                output_dir = Path(f"gifd_results/{Path(exp_dir).name}")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                save_attack_results(
                    results, output_dir,
                    f"gifd_{self.gan_type}_r{round_num}_{client_id}"
                )
                
                return results
                
            else:
                print("❌ No reconstruction output generated")
                return None
                
        except Exception as e:
            print(f"❌ Attack failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def attack_experiment(self, exp_dir, rounds=None, clients=None):
        """
        Run GIFD attack on multiple rounds and clients
        
        Args:
            exp_dir: Experiment directory
            rounds: List of rounds to attack (None for all)
            clients: List of clients to attack (None for all)
            
        Returns:
            list: All attack results
        """
        from run_grad_inv import discover_experiments, select_experiment
        
        # Find experiment info
        experiments = discover_experiments()
        exp_name = Path(exp_dir).name
        experiment = select_experiment(exp_name, experiments)
        
        # Determine targets
        target_rounds = rounds if rounds else experiment['rounds']
        target_clients = clients if clients else experiment['clients']
        
        print(f"\n📋 GIFD Attack Campaign")
        print(f"   Experiment: {exp_name}")
        print(f"   Rounds: {target_rounds}")
        print(f"   Clients: {target_clients[:3]}{'...' if len(target_clients) > 3 else ''}")
        print(f"   Total attacks: {len(target_rounds) * len(target_clients)}")
        
        all_results = []
        successful_attacks = 0
        
        for round_num in target_rounds:
            for client_id in target_clients:
                result = self.attack_client(exp_dir, round_num, client_id)
                if result:
                    all_results.append(result)
                    if result['psnr'] > 20:
                        successful_attacks += 1
        
        if all_results:
            avg_psnr = sum(r['psnr'] for r in all_results) / len(all_results)
            avg_ssim = sum(r['ssim'] for r in all_results) / len(all_results)
            
            print(f"\n📊 GIFD Attack Summary")
            print(f"   Total attacks: {len(all_results)}")
            print(f"   Successful (>20dB): {successful_attacks}")
            print(f"   Average PSNR: {avg_psnr:.2f} dB")
            print(f"   Average SSIM: {avg_ssim:.4f}")
            
            # Generate summary report
            self._generate_summary_report(all_results, exp_name)
        
        return all_results
    
    def _generate_summary_report(self, results, exp_name):
        """Generate summary report for GIFD attacks"""
        output_dir = Path(f"gifd_results/{exp_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"gifd_{self.gan_type}_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"GIFD Attack Summary Report\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"GAN Type: {self.gan_type}\n")
            f.write(f"Configuration: {self.config_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Results by Round and Client:\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{'Round':<10} {'Client':<15} {'PSNR (dB)':<12} {'SSIM':<10} {'Time (s)':<10}\n")
            f.write(f"{'-'*60}\n")
            
            for r in results:
                f.write(f"{r['round']:<10} {r['client_id']:<15} "
                       f"{r['psnr']:<12.2f} {r['ssim']:<10.4f} "
                       f"{r['reconstruction_time']:<10.2f}\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"Summary Statistics:\n")
            f.write(f"  Total Attacks: {len(results)}\n")
            f.write(f"  Average PSNR: {sum(r['psnr'] for r in results)/len(results):.2f} dB\n")
            f.write(f"  Average SSIM: {sum(r['ssim'] for r in results)/len(results):.4f}\n")
            f.write(f"  Success Rate (>20dB): {sum(1 for r in results if r['psnr'] > 20)/len(results)*100:.1f}%\n")
        
        print(f"📝 Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='GIFD Attack Runner - Generative Gradient Inversion with Feature Domain Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GAN Types:
  stylegan2-ada - StyleGAN2-ADA with CIFAR-10 pkl checkpoint (recommended)
  dcgan         - DCGAN optimized for CIFAR-10 (needs training)
  stylegan2     - StyleGAN2 standard (needs FFHQ checkpoint)
  biggan        - BigGAN for ImageNet-scale attacks
  none          - GAN-free optimization (baseline)

Attack Configurations:
  quick_test - Fast testing (1 restart, 2K iterations)
  default    - Balanced (2 restarts, 4K iterations)
  aggressive - Strong (4 restarts, 8K iterations)

Examples:
  %(prog)s --gan dcgan                            # DCGAN for CIFAR-10
  %(prog)s --gan biggan --config aggressive       # Strong BigGAN attack
  %(prog)s --experiment cifar10_classic_c3_r3     # Specific experiment
  %(prog)s --rounds 1 2 --clients c0_1 c0_2      # Specific targets

Note: For CIFAR-10, use --gan dcgan as it's specifically trained on CIFAR-10.
      StyleGAN2 requires FFHQ checkpoint which is for faces, not CIFAR-10.
        """
    )
    
    # GAN and configuration
    parser.add_argument('--gan', choices=['dcgan', 'stylegan2', 'stylegan2-ada', 'biggan', 'none'], 
                       default='stylegan2-ada',
                       help='GAN type to use (default: %(default)s for CIFAR-10 with pkl)')
    parser.add_argument('--config', choices=list(GIFD_CONFIGS.keys()),
                       default='default',
                       help='Attack configuration (default: %(default)s)')
    
    # Target selection (reuse from run_grad_inv.py)
    parser.add_argument('--experiment', '-e', type=str,
                       help='Experiment name or number')
    parser.add_argument('--rounds', '-r', type=int, nargs='+',
                       help='Specific rounds to attack')
    parser.add_argument('--clients', '-c', type=str, nargs='+',
                       help='Specific clients to attack')
    
    # Execution options
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                       default='auto',
                       help='Computation device (default: %(default)s)')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments and exit')
    
    args = parser.parse_args()
    
    print("🎯 GIFD Attack Runner")
    print("="*50)
    
    # Import discovery functions from run_grad_inv
    from run_grad_inv import discover_experiments, print_available_experiments, select_experiment
    
    # Discover experiments
    experiments = discover_experiments()
    
    if args.list:
        print_available_experiments(experiments)
        return 0
    
    if not experiments:
        print("❌ No experiments with gradient data found!")
        return 1
    
    try:
        # Select experiment
        selected_exp = select_experiment(args.experiment, experiments)
        print(f"🎯 Selected: {selected_exp['name']}")
        print(f"   GAN Type: {args.gan}")
        print(f"   Config: {args.config}")
        
        # Check for GAN model files if needed
        if args.gan == 'stylegan2':
            stylegan_path = Path(GIFD_PATH) / 'inversefed' / 'genmodels' / 'stylegan2_io' / 'stylegan2-ffhq-config-f.pt'
            if not stylegan_path.exists():
                print(f"\n⚠️ StyleGAN2 checkpoint not found!")
                print(f"   Please download it:")
                print(f"   gdown --id 1JCBiKY_yUixTa6F1eflABL88T4cii2GR")
                print(f"   mv stylegan2-ffhq-config-f.pt {stylegan_path.parent}/")
                return 1
        
        # Initialize attacker
        attacker = GIFDAttacker(
            gan_type=args.gan,
            config=args.config,
            device=args.device
        )
        
        # Run attacks
        results = attacker.attack_experiment(
            selected_exp['path'],
            rounds=args.rounds,
            clients=args.clients
        )
        
        if results:
            print(f"\n✅ GIFD Attack Complete!")
            print(f"   Results saved to: gifd_results/{selected_exp['name']}/")
        else:
            print(f"\n❌ No successful attacks")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())