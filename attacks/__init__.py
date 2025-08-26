"""
Attack framework for privacy-preserving federated learning evaluation
"""

from .gradient_inversion import GradientInversionAttacker
from .attack_configs import ATTACK_CONFIGS

__all__ = ['GradientInversionAttacker', 'ATTACK_CONFIGS']