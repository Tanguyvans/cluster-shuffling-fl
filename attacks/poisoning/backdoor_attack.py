import torch
import numpy as np
from typing import Dict, Any, Tuple
from .base_poisoning_attack import BasePoisoningAttack


class BackdoorAttack(BasePoisoningAttack):
    """
    Backdoor poisoning attack.
    
    This attack injects backdoor triggers into training data to create
    a hidden functionality that can be activated later with specific inputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trigger_type = config.get('trigger_type', 'pixel_pattern')
        self.trigger_size = config.get('trigger_size', 3)  # Size of trigger pattern
        self.trigger_position = config.get('trigger_position', 'bottom_right')
        self.trigger_value = config.get('trigger_value', 1.0)  # Trigger pixel intensity
        self.backdoor_label = config.get('backdoor_label', self.target_class)
        self.poison_all_classes = config.get('poison_all_classes', True)
        
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject backdoor triggers into training data.
        
        Args:
            data: Training data tensor [N, C, H, W]
            labels: Training labels tensor [N]
            
        Returns:
            Tuple of (poisoned_data, poisoned_labels)
        """
        if not self.should_attack():
            return data, labels
            
        poisoned_data = data.clone()
        poisoned_labels = labels.clone()
        
        num_samples = len(data)
        num_to_poison = int(num_samples * self.attack_intensity)
        
        if num_to_poison == 0:
            return data, labels
            
        # Select samples to poison
        if self.poison_all_classes:
            # Poison samples from all classes
            poison_indices = torch.randperm(num_samples)[:num_to_poison]
        else:
            # Only poison samples from non-target classes
            non_target_indices = (labels != self.backdoor_label).nonzero(as_tuple=True)[0]
            if len(non_target_indices) == 0:
                return data, labels
            poison_indices = non_target_indices[torch.randperm(len(non_target_indices))[:num_to_poison]]
            
        # Apply backdoor triggers
        for idx in poison_indices:
            poisoned_data[idx] = self._apply_trigger(poisoned_data[idx])
            poisoned_labels[idx] = self.backdoor_label
            
        return poisoned_data, poisoned_labels
        
    def _apply_trigger(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply backdoor trigger to a single sample."""
        triggered_sample = sample.clone()
        
        if self.trigger_type == 'pixel_pattern':
            triggered_sample = self._apply_pixel_pattern(triggered_sample)
        elif self.trigger_type == 'square':
            triggered_sample = self._apply_square_trigger(triggered_sample)
        elif self.trigger_type == 'cross':
            triggered_sample = self._apply_cross_trigger(triggered_sample)
        elif self.trigger_type == 'random_noise':
            triggered_sample = self._apply_noise_trigger(triggered_sample)
        else:
            raise ValueError(f"Unknown trigger_type: {self.trigger_type}. Available: 'pixel_pattern', 'square', 'cross', 'random_noise'")
            
        return triggered_sample
        
    def _apply_pixel_pattern(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply simple pixel pattern trigger."""
        channels, height, width = sample.shape
        
        if self.trigger_position == 'bottom_right':
            start_h = height - self.trigger_size
            start_w = width - self.trigger_size
        elif self.trigger_position == 'top_left':
            start_h = 0
            start_w = 0
        elif self.trigger_position == 'center':
            start_h = height // 2 - self.trigger_size // 2
            start_w = width // 2 - self.trigger_size // 2
        else:
            # Random position
            start_h = np.random.randint(0, height - self.trigger_size + 1)
            start_w = np.random.randint(0, width - self.trigger_size + 1)
            
        # Apply trigger pattern (simple square)
        sample[:, start_h:start_h+self.trigger_size, start_w:start_w+self.trigger_size] = self.trigger_value
        
        return sample
        
    def _apply_square_trigger(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply square trigger pattern."""
        channels, height, width = sample.shape
        
        # Position square in bottom right
        start_h = height - self.trigger_size - 1
        start_w = width - self.trigger_size - 1
        end_h = start_h + self.trigger_size
        end_w = start_w + self.trigger_size
        
        # Create white square with black border
        sample[:, start_h:end_h, start_w:end_w] = self.trigger_value
        sample[:, start_h, start_w:end_w] = 0  # Top border
        sample[:, end_h-1, start_w:end_w] = 0  # Bottom border
        sample[:, start_h:end_h, start_w] = 0  # Left border
        sample[:, start_h:end_h, end_w-1] = 0  # Right border
        
        return sample
        
    def _apply_cross_trigger(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply cross-shaped trigger."""
        channels, height, width = sample.shape
        
        # Position cross in bottom right
        center_h = height - self.trigger_size
        center_w = width - self.trigger_size
        
        # Vertical line of cross
        for i in range(self.trigger_size):
            if center_h - i // 2 >= 0 and center_h + i // 2 < height:
                sample[:, center_h - i // 2:center_h + i // 2 + 1, center_w] = self.trigger_value
                
        # Horizontal line of cross
        for i in range(self.trigger_size):
            if center_w - i // 2 >= 0 and center_w + i // 2 < width:
                sample[:, center_h, center_w - i // 2:center_w + i // 2 + 1] = self.trigger_value
                
        return sample
        
    def _apply_noise_trigger(self, sample: torch.Tensor) -> torch.Tensor:
        """Apply random noise trigger in specific region."""
        channels, height, width = sample.shape
        
        # Add noise to bottom right corner
        start_h = height - self.trigger_size
        start_w = width - self.trigger_size
        
        noise = torch.randn(channels, self.trigger_size, self.trigger_size) * 0.5 + 0.5
        noise = torch.clamp(noise, 0, 1)
        
        sample[:, start_h:start_h+self.trigger_size, start_w:start_w+self.trigger_size] = noise
        
        return sample
        
        
    def poison_gradients(self, gradients: Dict[str, torch.Tensor],
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Backdoor attack primarily works through data poisoning.
        Can optionally enhance gradients to strengthen backdoor.
        
        Returns:
            Original gradients or slightly modified ones
        """
        if not self.should_attack():
            return gradients
            
        # Optional: slightly enhance gradients related to backdoor features
        enhance_strength = 0.1  # Subtle enhancement
        poisoned_gradients = {}
        
        for param_name, grad_tensor in gradients.items():
            if 'conv' in param_name.lower() or 'classifier' in param_name.lower():
                # Slightly enhance gradients for layers that might learn backdoor features
                enhanced_grad = grad_tensor * (1 + enhance_strength * self.attack_intensity)
                poisoned_gradients[param_name] = enhanced_grad
            else:
                poisoned_gradients[param_name] = grad_tensor
                
        return poisoned_gradients
        
    def create_backdoor_test_data(self, clean_data: torch.Tensor, 
                                 clean_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create test data with backdoor triggers for evaluation.
        
        Args:
            clean_data: Clean test data
            clean_labels: Clean test labels
            
        Returns:
            Tuple of (triggered_data, target_labels)
        """
        triggered_data = clean_data.clone()
        target_labels = torch.full_like(clean_labels, self.backdoor_label)
        
        # Apply triggers to all test samples
        for i in range(len(triggered_data)):
            triggered_data[i] = self._apply_trigger(triggered_data[i])
            
        return triggered_data, target_labels
        
    def get_attack_info(self) -> Dict[str, Any]:
        """Get detailed attack information."""
        info = super().get_attack_info()
        info.update({
            'trigger_type': self.trigger_type,
            'trigger_size': self.trigger_size,
            'trigger_position': self.trigger_position,
            'trigger_value': self.trigger_value,
            'backdoor_label': self.backdoor_label,
            'poison_all_classes': self.poison_all_classes
        })
        return info


# Register the attack with the factory
from .attack_factory import AttackFactory
AttackFactory.register_attack('backdoor', BackdoorAttack)
AttackFactory.register_attack('backdoor_poisoning', BackdoorAttack)