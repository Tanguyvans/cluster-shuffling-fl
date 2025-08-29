import torch
from typing import Dict, Any, Tuple
from .base_poisoning_attack import BasePoisoningAttack


class SignFlippingAttack(BasePoisoningAttack):
    """
    Sign flipping poisoning attack.
    
    This attack flips the signs of gradient updates to disrupt convergence.
    Based on "Byzantine-Robust Stochastic Aggregation Methods".
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.flip_strategy = config.get('flip_strategy', 'all')  # 'all', 'random', 'selective'
        self.target_layers = config.get('target_layers', None)  # None = all layers
        self.flip_probability = config.get('flip_probability', self.attack_intensity)
        self.magnitude_scaling = config.get('magnitude_scaling', 1.0)  # Scale flipped gradients
        
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sign flipping doesn't modify data directly - it works on gradients only.
        
        Returns:
            Original data and labels unchanged
        """
        # Signal that no data poisoning occurred
        return data, labels
        
    def poison_gradients(self, gradients: Dict[str, torch.Tensor],
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Flip signs of gradient updates.
        
        Args:
            gradients: Dictionary of gradient tensors
            model_state: Current model state dictionary
            round_info: Information about current round
            
        Returns:
            Dictionary of sign-flipped gradient tensors
        """
        if not self.should_attack():
            return gradients
            
        poisoned_gradients = {}
        
        for param_name, grad_tensor in gradients.items():
            if self._should_poison_layer(param_name):
                poisoned_gradients[param_name] = self._flip_gradient_signs(
                    grad_tensor, param_name)
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
        
    def _flip_gradient_signs(self, gradient: torch.Tensor, param_name: str) -> torch.Tensor:
        """Flip signs of gradient tensor based on strategy."""
        if self.flip_strategy == 'all':
            # BLADES-style: Complete sign flip with optional magnitude scaling
            flipped_gradient = -gradient * self.magnitude_scaling
            
        elif self.flip_strategy == 'random':
            # Randomly flip signs based on flip_probability
            flip_mask = torch.rand_like(gradient) < self.flip_probability
            flipped_gradient = gradient.clone()
            flipped_gradient[flip_mask] = -flipped_gradient[flip_mask] * self.magnitude_scaling
            
        elif self.flip_strategy == 'selective':
            # Flip signs of largest magnitude gradients (more sophisticated)
            flipped_gradient = self._selective_sign_flip(gradient)
            
        else:
            raise ValueError(f"Unknown flip_strategy: {self.flip_strategy}. Available: 'all', 'random', 'selective'")
            
        return flipped_gradient
        
    def _selective_sign_flip(self, gradient: torch.Tensor) -> torch.Tensor:
        """Flip signs of gradients with largest magnitudes."""
        flipped_gradient = gradient.clone()
        flat_grad = gradient.flatten()
        
        # Find indices of top gradients by magnitude
        num_to_flip = int(len(flat_grad) * self.flip_probability)
        if num_to_flip == 0:
            return gradient
            
        _, top_indices = torch.topk(torch.abs(flat_grad), num_to_flip)
        
        # Create flip mask
        flip_mask = torch.zeros_like(flat_grad, dtype=torch.bool)
        flip_mask[top_indices] = True
        flip_mask = flip_mask.reshape(gradient.shape)
        
        flipped_gradient[flip_mask] = -flipped_gradient[flip_mask] * self.magnitude_scaling
        return flipped_gradient
        
        
    def get_attack_info(self) -> Dict[str, Any]:
        """Get detailed attack information."""
        info = super().get_attack_info()
        info.update({
            'flip_strategy': self.flip_strategy,
            'target_layers': self.target_layers,
            'flip_probability': self.flip_probability,
            'magnitude_scaling': self.magnitude_scaling
        })
        return info


# Register the attack with the factory
from .attack_factory import AttackFactory
AttackFactory.register_attack('signflip', SignFlippingAttack)
AttackFactory.register_attack('sign_flipping', SignFlippingAttack)