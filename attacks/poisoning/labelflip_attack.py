import torch
import numpy as np
from typing import Dict, Any, Tuple
from .base_poisoning_attack import BasePoisoningAttack


class LabelFlippingAttack(BasePoisoningAttack):
    """
    Label flipping poisoning attack.
    
    This attack flips labels of training samples to a target class or random classes.
    Can perform targeted attacks (flip specific class to target) or random attacks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.flip_type = config.get('flip_type', 'targeted')  # 'targeted', 'random', 'all_to_one'
        self.source_class = config.get('source_class', None)  # Class to flip from (None = all classes)
        self.num_classes = config.get('num_classes', 10)
        
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Poison training data by flipping labels.
        
        Args:
            data: Training data tensor
            labels: Training labels tensor
            
        Returns:
            Tuple of (data, poisoned_labels)
        """
        if not self.should_attack():
            return data, labels
            
        poisoned_labels = labels.clone()
        num_samples = len(labels)
        num_to_poison = int(num_samples * self.attack_intensity)
        
        if num_to_poison == 0:
            return data, labels
            
        if self.flip_type == 'targeted':
            return self._targeted_label_flip(data, poisoned_labels, num_to_poison)
        elif self.flip_type == 'random':
            return self._random_label_flip(data, poisoned_labels, num_to_poison)
        elif self.flip_type == 'all_to_one':
            return self._all_to_one_flip(data, poisoned_labels, num_to_poison)
        else:
            raise ValueError(f"Unknown flip_type: {self.flip_type}")
            
    def _targeted_label_flip(self, data: torch.Tensor, labels: torch.Tensor, 
                           num_to_poison: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flip labels from source class to target class."""
        original_labels = labels.clone()
        
        if self.source_class is not None:
            # Flip specific source class to target class
            source_indices = (labels == self.source_class).nonzero(as_tuple=True)[0]
            if len(source_indices) == 0:
                return data, labels
            num_to_poison = min(num_to_poison, len(source_indices))
            poison_indices = source_indices[torch.randperm(len(source_indices))[:num_to_poison]]
        else:
            # Flip any non-target class samples to target class
            non_target_indices = (labels != self.target_class).nonzero(as_tuple=True)[0]
            if len(non_target_indices) == 0:
                return data, labels
            num_to_poison = min(num_to_poison, len(non_target_indices))
            poison_indices = non_target_indices[torch.randperm(len(non_target_indices))[:num_to_poison]]
            
        labels[poison_indices] = self.target_class
        self._log_label_changes(original_labels, labels, poison_indices, "targeted")
        return data, labels
        
    def _random_label_flip(self, data: torch.Tensor, labels: torch.Tensor,
                          num_to_poison: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flip labels to random classes."""
        original_labels = labels.clone()
        num_to_poison = min(num_to_poison, len(labels))
        poison_indices = torch.randperm(len(labels))[:num_to_poison]
        
        for idx in poison_indices:
            current_label = labels[idx].item()
            
            # Select random class different from current
            available_labels = list(range(self.num_classes))
            if current_label in available_labels:
                available_labels.remove(current_label)
            
            if available_labels:
                new_label = np.random.choice(available_labels)
                labels[idx] = new_label
        
        self._log_label_changes(original_labels, labels, poison_indices, "random")        
        return data, labels
        
    def _all_to_one_flip(self, data: torch.Tensor, labels: torch.Tensor,
                        num_to_poison: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flip all poisoned samples to target class."""
        original_labels = labels.clone()
        poison_indices = torch.randperm(len(labels))[:num_to_poison]
        labels[poison_indices] = self.target_class
        self._log_label_changes(original_labels, labels, poison_indices, "all_to_one")
        return data, labels
    
    def _log_label_changes(self, original_labels: torch.Tensor, poisoned_labels: torch.Tensor,
                          poison_indices: torch.Tensor, attack_type: str):
        """
        Log actual label changes in batch format.
        
        Args:
            original_labels: Original labels before poisoning
            poisoned_labels: Labels after poisoning
            poison_indices: Indices of poisoned samples
            attack_type: Type of label flipping attack
        """
        if len(poison_indices) == 0:
            return
            
        print(f"\n[LabelFlip] 🎯 {attack_type.upper()} attack applied:")
        print(f"[LabelFlip] 📊 Poisoned {len(poison_indices)}/{len(original_labels)} samples")
        
        # Show label changes for each poisoned sample
        label_changes = []
        for idx in poison_indices:
            original = original_labels[idx].item()
            poisoned = poisoned_labels[idx].item()
            label_changes.append(f"{original}→{poisoned}")
        
        # Display changes in compact batch format
        print(f"[LabelFlip] 🔄 Label changes: [{', '.join(label_changes)}]")
        
        # Show class distribution summary
        original_counts = torch.bincount(original_labels[poison_indices], minlength=self.num_classes)
        poisoned_counts = torch.bincount(poisoned_labels[poison_indices], minlength=self.num_classes)
        
        print(f"[LabelFlip] 📈 Class distribution (poisoned samples only):")
        print(f"  └─ Before: {dict(enumerate(original_counts.tolist()))}")
        print(f"  └─ After:  {dict(enumerate(poisoned_counts.tolist()))}")
        
    def poison_gradients(self, gradients: Dict[str, torch.Tensor],
                        model_state: Dict[str, torch.Tensor],
                        round_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Label flipping doesn't modify gradients directly - poisoning happens at data level.
        
        Returns:
            Original gradients unchanged
        """
        return gradients


# Register the attack with the factory
from .attack_factory import AttackFactory
AttackFactory.register_attack('labelflip', LabelFlippingAttack)
AttackFactory.register_attack('label_flipping', LabelFlippingAttack)