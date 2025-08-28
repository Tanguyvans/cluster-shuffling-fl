import torch
import numpy as np
from typing import Dict, Any, Tuple
from .base_poisoning_attack import BasePoisoningAttack
from .attack_factory import AttackFactory


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
                # Generate pure noise with same shape as gradient
                noise_mean = self.attack_intensity * 0.5  # Scale mean by attack intensity  
                noise_std = self.noise_std
                
                if self.noise_type == 'gaussian':
                    # BLADES uses torch.normal(mean, std) - pure noise replacement
                    pure_noise = torch.normal(
                        mean=noise_mean, 
                        std=noise_std, 
                        size=grad_tensor.shape,
                        device=grad_tensor.device,
                        dtype=grad_tensor.dtype
                    )
                elif self.noise_type == 'uniform':
                    pure_noise = torch.rand(
                        grad_tensor.shape,
                        device=grad_tensor.device,
                        dtype=grad_tensor.dtype
                    ) * (2 * noise_std) - noise_std + noise_mean
                else:
                    # Fallback to original method for other noise types
                    pure_noise = self._add_noise_to_gradient(grad_tensor, model_state.get(param_name, None))
                
                poisoned_gradients[param_name] = pure_noise
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
        
    def _add_noise_to_gradient(self, gradient: torch.Tensor, 
                              param_value: torch.Tensor = None) -> torch.Tensor:
        """Add noise to a gradient tensor."""
        if self.adaptive_noise and param_value is not None:
            # Scale noise based on parameter magnitude
            param_magnitude = torch.std(param_value)
            effective_noise_std = self.noise_std * param_magnitude
        else:
            effective_noise_std = self.noise_std
            
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(gradient) * effective_noise_std
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(gradient) - 0.5) * 2 * effective_noise_std
        elif self.noise_type == 'laplacian':
            noise = torch.tensor(np.random.laplace(0, effective_noise_std, gradient.shape),
                               dtype=gradient.dtype, device=gradient.device)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")
            
        return gradient + noise
        
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
AttackFactory.register_attack('noise', NoiseAttack)
AttackFactory.register_attack('noise_injection', NoiseAttack)