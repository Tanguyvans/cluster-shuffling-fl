import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BasePoisoningAttack(ABC):
    """Base class for all poisoning attacks in federated learning."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the poisoning attack.
        
        Args:
            config: Attack configuration dictionary
        """
        self.config: Dict[str, Any] = config
        self.attack_intensity: float = config.get('attack_intensity', 0.1)
        self.target_class: int = config.get('target_class', 0)
        self.round_number: int = 0
        
    def set_round(self, round_number: int) -> None:
        """Update the current round number."""
        self.round_number = round_number
        
    @abstractmethod
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Poison training data.
        
        Args:
            data: Training data tensor
            labels: Training labels tensor
            
        Returns:
            Tuple of (poisoned_data, poisoned_labels)
        """
        pass
        
    @abstractmethod
    def poison_gradients(self, gradients: Dict[str, torch.Tensor], 
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Poison gradient updates.
        
        Args:
            gradients: Dictionary of gradient tensors
            model_state: Current model state dictionary
            round_info: Information about current round
            
        Returns:
            Dictionary of poisoned gradient tensors
        """
        pass
        
    def poison_model_updates(self, model_updates: Dict[str, torch.Tensor],
                           round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Poison model parameter updates.
        
        Args:
            model_updates: Dictionary of model parameter updates
            round_info: Information about current round
            
        Returns:
            Dictionary of poisoned model updates
        """
        return self.poison_gradients(model_updates, {}, round_info)
        
    def should_attack(self, round_number: Optional[int] = None) -> bool:
        """
        Determine if attack should be applied in current round.
        
        Args:
            round_number: Current round number (uses self.round_number if None)
            
        Returns:
            Boolean indicating if attack should be applied
        """
        round_num = round_number if round_number is not None else self.round_number
        
        attack_rounds = self.config.get('attack_rounds', None)
        if attack_rounds is not None:
            return round_num in attack_rounds
            
        attack_frequency = self.config.get('attack_frequency', 1.0)
        return np.random.random() < attack_frequency
        
    def get_attack_info(self) -> Dict[str, Any]:
        """
        Get information about the attack configuration.
        
        Returns:
            Dictionary with attack information
        """
        return {
            'attack_type': self.__class__.__name__,
            'attack_intensity': self.attack_intensity,
            'target_class': self.target_class,
            'config': self.config
        }
        
    def _add_noise_to_tensor(self, tensor: torch.Tensor, 
                           noise_scale: Optional[float] = None) -> torch.Tensor:
        """
        Helper method to add Gaussian noise to tensor.
        
        Args:
            tensor: Input tensor
            noise_scale: Scale of noise (uses attack_intensity if None)
            
        Returns:
            Tensor with added noise
        """
        if noise_scale is None:
            noise_scale = self.attack_intensity
            
        noise = torch.randn_like(tensor) * noise_scale
        return tensor + noise
        
    def _flip_tensor_signs(self, tensor: torch.Tensor,
                          flip_probability: Optional[float] = None) -> torch.Tensor:
        """
        Helper method to flip signs of tensor elements.
        
        Args:
            tensor: Input tensor
            flip_probability: Probability of flipping each element
            
        Returns:
            Tensor with flipped signs
        """
        if flip_probability is None:
            flip_probability = self.attack_intensity
            
        flip_mask = torch.rand_like(tensor) < flip_probability
        flipped_tensor = tensor.clone()
        flipped_tensor[flip_mask] *= -1
        return flipped_tensor
        
    def _calculate_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Helper method to calculate basic tensor statistics.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Dictionary with tensor statistics
        """
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'norm': tensor.norm().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'numel': tensor.numel()
        }