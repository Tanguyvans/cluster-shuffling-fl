from typing import Dict, Any, Type
from .base_poisoning_attack import BasePoisoningAttack


class AttackFactory:
    """Factory class for creating poisoning attacks."""
    
    _attacks: Dict[str, Type[BasePoisoningAttack]] = {}
    
    @classmethod
    def register_attack(cls, name: str, attack_class: Type[BasePoisoningAttack]):
        """
        Register a new attack type.
        
        Args:
            name: Name of the attack
            attack_class: Attack class to register
        """
        cls._attacks[name.lower()] = attack_class
        
    @classmethod
    def create_attack(cls, attack_type: str, config: Dict[str, Any]) -> BasePoisoningAttack:
        """
        Create an attack instance.
        
        Args:
            attack_type: Type of attack to create
            config: Attack configuration
            
        Returns:
            Attack instance
            
        Raises:
            ValueError: If attack type is not registered
        """
        attack_type = attack_type.lower()
        if attack_type not in cls._attacks:
            available_attacks = list(cls._attacks.keys())
            raise ValueError(f"Unknown attack type: {attack_type}. "
                           f"Available attacks: {available_attacks}")
                           
        return cls._attacks[attack_type](config)
        
    @classmethod
    def list_available_attacks(cls) -> list:
        """
        List all available attack types.
        
        Returns:
            List of available attack type names
        """
        return list(cls._attacks.keys())
        
    @classmethod
    def get_attack_info(cls, attack_type: str) -> Dict[str, Any]:
        """
        Get information about an attack type.
        
        Args:
            attack_type: Type of attack
            
        Returns:
            Dictionary with attack information
        """
        attack_type = attack_type.lower()
        if attack_type not in cls._attacks:
            raise ValueError(f"Unknown attack type: {attack_type}")
            
        attack_class = cls._attacks[attack_type]
        return {
            'name': attack_type,
            'class': attack_class.__name__,
            'module': attack_class.__module__,
            'docstring': attack_class.__doc__
        }