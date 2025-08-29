import torch
from typing import Dict, Any, Tuple
from .base_poisoning_attack import BasePoisoningAttack


class ALIEAttack(BasePoisoningAttack):
    """
    ALIE (A Little Is Enough) attack.
    
    This attack circumvents defenses by making minimal changes that are
    difficult to detect. Based on "A little is enough: Circumventing defenses 
    for distributed learning".
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.deviation_type = config.get('deviation_type', 'sign')  # 'sign', 'std', 'mean'
        self.aggregation_type = config.get('aggregation_type', 'mean')  # Expected aggregation method
        self.epsilon = config.get('epsilon', self.attack_intensity)  # Small deviation parameter
        self.num_malicious = config.get('num_malicious', 1)  # Number of malicious clients
        self.historical_updates = []  # Store previous updates for analysis
        
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ALIE primarily works at gradient level, minimal data poisoning.
        
        Returns:
            Original data and labels unchanged
        """
        return data, labels
        
    def poison_gradients(self, gradients: Dict[str, torch.Tensor],
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Apply ALIE attack to gradient updates.
        
        Args:
            gradients: Dictionary of gradient tensors
            model_state: Current model state dictionary
            round_info: Information about current round
            
        Returns:
            Dictionary of ALIE-poisoned gradient tensors
        """
        if not self.should_attack():
            return gradients
            
        # Store current gradients for historical analysis
        self.historical_updates.append({k: v.clone() for k, v in gradients.items()})
        
        # Keep only recent history to avoid memory issues
        if len(self.historical_updates) > 10:
            self.historical_updates.pop(0)
            
        poisoned_gradients = {}
        
        for param_name, grad_tensor in gradients.items():
            poisoned_gradients[param_name] = self._apply_alie_deviation(
                grad_tensor, param_name, round_info)
                
        return poisoned_gradients
        
    def _apply_alie_deviation(self, gradient: torch.Tensor, param_name: str,
                             round_info: Dict[str, Any]) -> torch.Tensor:
        """Apply minimal deviation based on ALIE strategy."""
        
        if self.deviation_type == 'sign':
            return self._sign_based_deviation(gradient)
        elif self.deviation_type == 'std':
            return self._std_based_deviation(gradient)
        elif self.deviation_type == 'mean':
            return self._mean_based_deviation(gradient)
        else:
            raise ValueError(f"Unknown deviation_type: {self.deviation_type}. Available: 'sign', 'std', 'mean'")
            
    def _sign_based_deviation(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply sign-based minimal deviation."""
        # Create small deviation in opposite direction of gradient
        deviation_magnitude = self.epsilon * torch.std(gradient)
        deviation_direction = -torch.sign(gradient)
        
        # Add small random component to avoid detection
        random_component = torch.randn_like(gradient) * (deviation_magnitude * 0.1)
        
        deviation = deviation_direction * deviation_magnitude + random_component
        return gradient + deviation
        
    def _std_based_deviation(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply standard deviation based minimal deviation."""
        grad_std = torch.std(gradient)
        grad_mean = torch.mean(gradient)
        
        # Create deviation that slightly shifts the distribution
        deviation = torch.randn_like(gradient) * (grad_std * self.epsilon)
        
        # Bias towards reducing gradient magnitude
        bias = -grad_mean * self.epsilon
        
        return gradient + deviation + bias
        
    def _mean_based_deviation(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply mean-based minimal deviation."""
        grad_mean = torch.mean(gradient)
        
        # Create deviation that shifts the mean slightly
        deviation = torch.full_like(gradient, -grad_mean * self.epsilon)
        
        # Add small random noise to mask the attack
        noise = torch.randn_like(gradient) * (torch.std(gradient) * self.epsilon * 0.1)
        
        return gradient + deviation + noise
        
        
    def _estimate_benign_updates(self) -> Dict[str, torch.Tensor]:
        """Estimate what benign updates might look like based on history."""
        if len(self.historical_updates) < 2:
            return {}
            
        # Simple estimation: average of recent updates
        estimated_updates = {}
        param_names = self.historical_updates[-1].keys()
        
        for param_name in param_names:
            recent_updates = [update[param_name] for update in self.historical_updates[-3:]]
            estimated_updates[param_name] = torch.stack(recent_updates).mean(dim=0)
            
        return estimated_updates
        
    def get_attack_info(self) -> Dict[str, Any]:
        """Get detailed attack information."""
        info = super().get_attack_info()
        info.update({
            'deviation_type': self.deviation_type,
            'aggregation_type': self.aggregation_type,
            'epsilon': self.epsilon,
            'num_malicious': self.num_malicious,
            'historical_updates_stored': len(self.historical_updates)
        })
        return info


# Register the attack with the factory
from .attack_factory import AttackFactory
AttackFactory.register_attack('alie', ALIEAttack)
AttackFactory.register_attack('a_little_is_enough', ALIEAttack)