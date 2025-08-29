import torch
import glob
import os
import time
from typing import Dict, Any, Tuple, Optional, List
from .base_poisoning_attack import BasePoisoningAttack


class IPMAttack(BasePoisoningAttack):
    """
    Inner Product Manipulation (IPM) attack.
    
    Manipulates gradients to disrupt aggregation while remaining undetected.
    Based on "Breaking byzantine-tolerant SGD by inner product manipulation".
    
    Supports two modes:
    - Standard IPM: Targets mean of benign gradients
    - Cross-cluster IPM: Targets other cluster's gradients (for cluster-shuffling defense)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.manipulation_strategy = config.get('manipulation_strategy', 'maximize_distance')
        self.target_level = config.get('target_level', 'client')  # client, cluster, cross_cluster
        self.lambda_param = config.get('lambda_param', self.attack_intensity)
        self.gradient_history: List[torch.Tensor] = []
        self.cross_cluster_attack_info: List[Dict] = []
        
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """IPM works at gradient level only."""
        return data, labels
        
    def poison_gradients(self, gradients: Dict[str, torch.Tensor],
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Apply IPM attack: Replace gradients with -λ * estimated_benign_mean.
        
        Args:
            gradients: Dictionary of gradient tensors
            model_state: Current model state dictionary
            round_info: Information about current round
            
        Returns:
            Dictionary of IPM-manipulated gradient tensors
        """
        if not self.should_attack():
            return gradients
        
        # Store gradient history for estimation
        self._update_gradient_history(gradients)
        
        # Apply IPM manipulation to each gradient
        poisoned_gradients = {}
        for param_name, grad_tensor in gradients.items():
            poisoned_gradients[param_name] = self._apply_ipm(grad_tensor)
                
        return poisoned_gradients
    
    def _update_gradient_history(self, gradients: Dict[str, torch.Tensor]):
        """Store flattened gradient history for benign estimation."""
        flattened_grads = []
        for grad_tensor in gradients.values():
            flattened_grads.append(grad_tensor.flatten())
        combined_grad = torch.cat(flattened_grads)
        
        # Keep last 5 gradient updates
        if len(self.gradient_history) >= 5:
            self.gradient_history.pop(0)
        self.gradient_history.append(combined_grad.clone().detach())
        
    def _apply_ipm(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Apply IPM transformation: -λ * mean(benign_gradients).
        
        Args:
            gradient: Current gradient tensor
            
        Returns:
            IPM-manipulated gradient
        """
        # Get benign gradient estimate based on target level
        if self.target_level == "cross_cluster":
            benign_mean = self._get_cross_cluster_target(gradient)
        else:
            benign_mean = self._get_standard_benign_mean(gradient)
        
        # Apply IPM formula: -λ * benign_mean
        return -self.lambda_param * benign_mean
    
    def _get_cross_cluster_target(self, current_gradient: torch.Tensor) -> torch.Tensor:
        """
        Cross-cluster IPM: Wait for and target other cluster's gradients.
        
        Args:
            current_gradient: Current gradient tensor
            
        Returns:
            Scaled mean of other cluster's gradients
        """
        print("[IPM] 🎯 Cross-cluster targeting mode activated")
        print("[IPM] ⏳ Waiting for other cluster gradients...")
        
        wait_timeout = 10
        start_time = time.time()
        check_interval = 0.5
        
        while (time.time() - start_time) < wait_timeout:
            # Look for recent gradient files from other clusters
            gradient_files = glob.glob("results/*/models/clients/round_*/*_gradients.pt")
            other_cluster_gradients = []
            
            for gradient_file in gradient_files[-20:]:  # Check most recent files
                try:
                    client_id = os.path.basename(gradient_file).split("_gradients.pt")[0]
                    
                    # Skip attacking client (c0_1)
                    if client_id == "c0_1":
                        continue
                    
                    # Load and extract matching gradient
                    gradient_data = torch.load(gradient_file, map_location='cpu')
                    if isinstance(gradient_data, dict) and 'gradients' in gradient_data:
                        for grad_tensor in gradient_data['gradients']:
                            if isinstance(grad_tensor, torch.Tensor) and grad_tensor.shape == current_gradient.shape:
                                other_cluster_gradients.append(grad_tensor)
                                break
                                
                except Exception:
                    continue
            
            # If we have enough gradients from other cluster(s), use them
            if len(other_cluster_gradients) >= 2:
                # Calculate mean of other cluster gradients
                other_cluster_mean = torch.stack(other_cluster_gradients[-3:]).mean(dim=0)
                
                # Scale up to compensate for cluster dilution
                scale_factor = 3.0
                scaled_target = scale_factor * other_cluster_mean
                
                # Log attack details
                self._log_cross_cluster_attack(other_cluster_mean, scale_factor, len(other_cluster_gradients))
                
                return scaled_target
                
            time.sleep(check_interval)
        
        print(f"[IPM] ⏰ Timeout after {wait_timeout}s, falling back to standard IPM")
        return self._get_standard_benign_mean(current_gradient)
    
    def _get_standard_benign_mean(self, current_gradient: torch.Tensor) -> torch.Tensor:
        """
        Standard IPM: Get mean of available benign gradients.
        
        Args:
            current_gradient: Current gradient tensor
            
        Returns:
            Mean of benign gradients or current gradient as fallback
        """
        # Find most recent round with gradient files
        results_dirs = glob.glob("results/*/models/clients/round_*")
        
        if not results_dirs:
            print("[IPM] No saved gradients found, using current gradient")
            return current_gradient
        
        # Extract round number from most recent directory
        latest_dir = sorted(results_dirs)[-1]
        round_part = latest_dir.split("/")[-1]
        
        if not round_part.startswith("round_"):
            return current_gradient
            
        current_round = int(round_part.split("_")[1])
        
        # Load benign gradients from current round
        benign_gradients = []
        gradient_files = glob.glob(f"results/*/models/clients/round_{current_round:03d}/*_gradients.pt")
        
        for gradient_file in gradient_files:
            try:
                client_id = os.path.basename(gradient_file).split("_gradients.pt")[0]
                
                # Skip malicious client
                if client_id == "c0_1":
                    continue
                
                # Load and extract matching gradient
                gradient_data = torch.load(gradient_file, map_location='cpu')
                if isinstance(gradient_data, dict) and 'gradients' in gradient_data:
                    for grad_tensor in gradient_data['gradients']:
                        if isinstance(grad_tensor, torch.Tensor) and grad_tensor.shape == current_gradient.shape:
                            benign_gradients.append(grad_tensor)
                            print(f"[IPM] Loaded gradient from benign client {client_id}")
                            break
                            
            except Exception as e:
                continue
        
        if benign_gradients:
            benign_mean = torch.stack(benign_gradients).mean(dim=0)
            print(f"[IPM] Using mean of {len(benign_gradients)} benign gradients")
            return benign_mean
        else:
            print("[IPM] No compatible benign gradients found, using current gradient")
            return current_gradient
    
    def _log_cross_cluster_attack(self, other_cluster_mean: torch.Tensor, 
                                  scale_factor: float, gradient_count: int):
        """Log cross-cluster attack details for analysis."""
        print(f"\n[IPM] 🎯 CROSS-CLUSTER ATTACK SUCCESSFUL!")
        print(f"[IPM] 📊 Target Analysis:")
        print(f"  └─ Other cluster gradient count: {gradient_count}")
        print(f"  └─ Target cluster mean norm: {torch.norm(other_cluster_mean).item():.6f}")
        print(f"  └─ After scaling (x{scale_factor}): {torch.norm(scale_factor * other_cluster_mean).item():.6f}")
        print(f"[IPM] ⚡ Attack Strategy:")
        print(f"  └─ Formula: attack_grad = -λ * scale * mean(other_cluster_grads)")
        print(f"  └─ Scale factor: {scale_factor} (compensates for cluster dilution)")
        print(f"  └─ Attack intensity (λ): {self.lambda_param}")
        print(f"  └─ Expected impact: ~{self.lambda_param * scale_factor:.1f}x other cluster")
        
        # Store attack info for JSON logging
        attack_info = {
            'attack_type': 'cross_cluster_ipm',
            'target_gradient_count': gradient_count,
            'target_norm': torch.norm(other_cluster_mean).item(),
            'scale_factor': scale_factor,
            'scaled_target_norm': torch.norm(scale_factor * other_cluster_mean).item(),
            'attack_intensity': self.lambda_param,
            'expected_cancellation_ratio': self.lambda_param * scale_factor,
            'parameter_shape': str(other_cluster_mean.shape)
        }
        self.cross_cluster_attack_info.append(attack_info)
        
    def get_attack_info(self) -> Dict[str, Any]:
        """Get detailed attack information including cross-cluster details."""
        info = super().get_attack_info()
        info.update({
            'manipulation_strategy': self.manipulation_strategy,
            'target_level': self.target_level,
            'lambda_param': self.lambda_param,
            'gradient_history_length': len(self.gradient_history)
        })
        
        # Add cross-cluster attack information if available
        if self.cross_cluster_attack_info:
            info.update({
                'cross_cluster_attacks': self.cross_cluster_attack_info,
                'total_cross_cluster_attacks': len(self.cross_cluster_attack_info),
                'attack_variant': 'cross_cluster_ipm',
                'cross_cluster_summary': {
                    'average_target_norm': sum(a['target_norm'] for a in self.cross_cluster_attack_info) / len(self.cross_cluster_attack_info),
                    'average_cancellation_ratio': sum(a['expected_cancellation_ratio'] for a in self.cross_cluster_attack_info) / len(self.cross_cluster_attack_info),
                    'scale_factor_used': self.cross_cluster_attack_info[0]['scale_factor'] if self.cross_cluster_attack_info else None
                }
            })
            
        return info


# Register the attack with the factory
from .attack_factory import AttackFactory
AttackFactory.register_attack('ipm', IPMAttack)
AttackFactory.register_attack('inner_product_manipulation', IPMAttack)