import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base_poisoning_attack import BasePoisoningAttack


class NoiseAttack(BasePoisoningAttack):
    """
    Noise injection poisoning attack.
    
    This attack adds Gaussian noise to model gradients/updates to disrupt
    the federated learning process. The noise can be applied to all parameters
    or selectively to specific layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.noise_type = config.get('noise_type', 'gaussian')  # 'gaussian', 'uniform', 'laplacian'
        self.noise_std = config.get('noise_std', self.attack_intensity)
        self.target_layers = config.get('target_layers', None)  # None = all layers
        self.adaptive_noise = config.get('adaptive_noise', False)  # Scale noise based on parameter magnitude
        
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to training data.
        
        Args:
            data: Training data tensor
            labels: Training labels tensor
            
        Returns:
            Tuple of (poisoned_data, labels)
        """
        if not self.should_attack():
            return data, labels
            
        poisoned_data = data.clone()
        
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(data) * self.noise_std
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(data) - 0.5) * 2 * self.noise_std
        elif self.noise_type == 'laplacian':
            noise = torch.tensor(np.random.laplace(0, self.noise_std, data.shape), 
                               dtype=data.dtype, device=data.device)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")
            
        poisoned_data += noise
        return poisoned_data, labels
        
    def poison_gradients(self, gradients: Dict[str, torch.Tensor],
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Replace gradient updates with noise (following BLADES implementation).
        
        Args:
            gradients: Dictionary of gradient tensors
            model_state: Current model state dictionary
            round_info: Information about current round
            
        Returns:
            Dictionary of poisoned gradient tensors
        """
        if not self.should_attack():
            return gradients
            
        poisoned_gradients = {}
        
        # BLADES-style: Replace gradients with pure noise instead of adding noise
        for param_name, grad_tensor in gradients.items():
            if self._should_poison_layer(param_name):
                poisoned_gradients[param_name] = self._generate_noise_tensor(
                    grad_tensor.shape, grad_tensor.device, grad_tensor.dtype)
            else:
                poisoned_gradients[param_name] = grad_tensor
                
        return poisoned_gradients
        
    def _should_poison_layer(self, param_name: str) -> bool:
        """Check if a parameter layer should be poisoned."""
        if self.target_layers is None:
            return True
            
        for target_layer in self.target_layers:
            if target_layer in param_name:
                return True
        return False
        
    def _generate_noise_tensor(self, shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate pure noise tensor with specified properties."""
        noise_mean = self.attack_intensity * 0.5
        noise_std = self.noise_std
        
        if self.noise_type == 'gaussian':
            return torch.normal(mean=noise_mean, std=noise_std, size=shape, device=device, dtype=dtype)
        elif self.noise_type == 'uniform':
            return torch.rand(shape, device=device, dtype=dtype) * (2 * noise_std) - noise_std + noise_mean
        elif self.noise_type == 'laplacian':
            noise_np = np.random.laplace(noise_mean, noise_std, shape)
            return torch.tensor(noise_np, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}. Available: 'gaussian', 'uniform', 'laplacian'")
        
    def get_attack_info(self) -> Dict[str, Any]:
        """Get detailed attack information."""
        info = super().get_attack_info()
        info.update({
            'noise_type': self.noise_type,
            'noise_std': self.noise_std,
            'target_layers': self.target_layers,
            'adaptive_noise': self.adaptive_noise
        })
        return info


# Register the attack with the factory
from .attack_factory import AttackFactory
AttackFactory.register_attack('noise', NoiseAttack)
AttackFactory.register_attack('noise_injection', NoiseAttack)