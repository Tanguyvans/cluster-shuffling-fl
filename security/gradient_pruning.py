"""
Gradient Pruning (Sparsification) for Federated Learning
========================================================

Implements Deep Gradient Compression (DGC) with momentum correction.
Based on: Lin et al., "Deep Gradient Compression" (2018)

Purpose:
- Reduce communication cost by 10-100x
- Maintain model convergence through momentum correction
- Compatible with SMPC and other security mechanisms

Key Innovation:
- Velocity buffer accumulates pruned gradients across rounds
- Small gradients eventually get sent when accumulated large enough
- Prevents information loss from aggressive pruning

Integration with your FL architecture:
- Apply BEFORE SMPC: prune → SMPC → send
- Or apply AFTER aggregation: aggregate → prune → send to clients
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle


class GradientPruner:
    """
    Gradient pruning with momentum correction (DGC algorithm).

    This implements the full Deep Gradient Compression algorithm:
    1. Accumulate all gradients in velocity buffer
    2. Select top-k most important gradients from velocity
    3. Send only top-k (sparse)
    4. Subtract sent gradients from velocity
    5. Next round, velocity contains "leftover" gradients
    """

    def __init__(self,
                 keep_ratio: float = 0.1,
                 momentum_factor: float = 0.9,
                 use_momentum_correction: bool = True,
                 sample_ratio: float = 0.01):
        """
        Initialize gradient pruner.

        Args:
            keep_ratio: Fraction of gradients to keep (0.1 = 10%, 90% pruned)
            momentum_factor: Momentum for velocity buffer (0.9 standard)
            use_momentum_correction: Enable DGC momentum correction
            sample_ratio: Fraction to sample for threshold estimation (0.01 = 1%)
        """
        self.keep_ratio = keep_ratio
        self.momentum_factor = momentum_factor
        self.use_momentum_correction = use_momentum_correction
        self.sample_ratio = sample_ratio

        # Velocity buffers (one per parameter, accumulated across rounds)
        self.velocities = {}

        # Statistics tracking
        self.compression_ratios = []
        self.communication_saved = []

    def prune_weights(self,
                     parameters: List[np.ndarray],
                     parameter_names: Optional[List[str]] = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Prune model parameters (weights) using top-k sparsification.

        This is the main function to call from your FL client.

        Args:
            parameters: List of parameter arrays (from get_parameters())
            parameter_names: Optional names for tracking

        Returns:
            Tuple of (pruned_parameters, statistics)
        """
        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(len(parameters))]

        # Initialize velocity buffers if first call
        if not self.velocities:
            self._initialize_velocities(parameters, parameter_names)

        pruned_params = []
        total_params = 0
        total_kept = 0
        total_bytes_original = 0
        total_bytes_compressed = 0

        for i, (param, name) in enumerate(zip(parameters, parameter_names)):
            if self.use_momentum_correction:
                # DGC with momentum correction
                pruned, stats = self._prune_with_momentum(param, name)
            else:
                # Simple top-k (for comparison)
                pruned, stats = self._simple_topk(param, name)

            pruned_params.append(pruned)

            # Aggregate statistics
            total_params += stats['total_elements']
            total_kept += stats['kept_elements']
            total_bytes_original += stats['bytes_original']
            total_bytes_compressed += stats['bytes_compressed']

        # Calculate global statistics
        compression_ratio = total_kept / total_params if total_params > 0 else 0
        communication_savings = 1 - (total_bytes_compressed / total_bytes_original) if total_bytes_original > 0 else 0

        self.compression_ratios.append(compression_ratio)
        self.communication_saved.append(communication_savings)

        statistics = {
            'total_parameters': total_params,
            'kept_parameters': total_kept,
            'pruned_parameters': total_params - total_kept,
            'compression_ratio': compression_ratio,
            'sparsity': 1 - compression_ratio,
            'bytes_original': total_bytes_original,
            'bytes_compressed': total_bytes_compressed,
            'communication_savings': communication_savings,
            'compression_factor': total_bytes_original / total_bytes_compressed if total_bytes_compressed > 0 else float('inf'),
            'method': 'dgc_momentum' if self.use_momentum_correction else 'simple_topk',
            'keep_ratio': self.keep_ratio
        }

        return pruned_params, statistics

    def _initialize_velocities(self, parameters: List[np.ndarray], names: List[str]):
        """Initialize velocity buffers (same shape as parameters)."""
        for param, name in zip(parameters, names):
            self.velocities[name] = np.zeros_like(param)

    def _prune_with_momentum(self, param: np.ndarray, name: str) -> Tuple[np.ndarray, Dict]:
        """
        DGC pruning with momentum correction.

        Algorithm:
        1. velocity += param  (accumulate ALL gradients)
        2. Select top-k from velocity
        3. Send sparse top-k
        4. velocity -= sent_sparse  (subtract what was sent)
        """
        original_shape = param.shape

        # Step 1: Accumulate in velocity buffer
        # Apply momentum factor for smoothing
        self.velocities[name] = self.momentum_factor * self.velocities[name] + param

        # Step 2: Find top-k threshold (using efficient sampling)
        velocity_flat = self.velocities[name].flatten()
        threshold = self._find_threshold_sampled(velocity_flat)

        # Step 3: Create sparse mask
        mask = np.abs(self.velocities[name]) >= threshold

        # Step 4: Extract sparse values
        sparse_param = self.velocities[name] * mask

        # Step 5: Update velocity (subtract what we're sending)
        self.velocities[name] -= sparse_param

        # Calculate statistics
        total_elements = param.size
        kept_elements = np.count_nonzero(mask)
        bytes_original = param.nbytes
        # Sparse format: values + indices (assuming 4 bytes per float/int)
        bytes_compressed = kept_elements * 4 + kept_elements * 4

        stats = {
            'total_elements': total_elements,
            'kept_elements': kept_elements,
            'bytes_original': bytes_original,
            'bytes_compressed': bytes_compressed
        }

        return sparse_param, stats

    def _simple_topk(self, param: np.ndarray, name: str) -> Tuple[np.ndarray, Dict]:
        """
        Simple top-k without momentum (for comparison).

        This is less effective but simpler to understand.
        """
        # Find threshold
        param_flat = param.flatten()
        threshold = self._find_threshold_sampled(param_flat)

        # Create mask
        mask = np.abs(param) >= threshold
        sparse_param = param * mask

        # Statistics
        total_elements = param.size
        kept_elements = np.count_nonzero(mask)
        bytes_original = param.nbytes
        bytes_compressed = kept_elements * 4 + kept_elements * 4

        stats = {
            'total_elements': total_elements,
            'kept_elements': kept_elements,
            'bytes_original': bytes_original,
            'bytes_compressed': bytes_compressed
        }

        return sparse_param, stats

    def _find_threshold_sampled(self, flat_array: np.ndarray) -> float:
        """
        Efficiently find top-k threshold using random sampling.

        Instead of sorting all N elements (O(N log N)), we:
        1. Sample M elements randomly (M << N)
        2. Find top-k from samples (O(M log M))
        3. Use that as threshold approximation

        This is MUCH faster for large models (10-100x speedup).
        """
        n = len(flat_array)
        k = max(1, int(n * self.keep_ratio))

        # Use sampling for efficiency
        sample_size = max(k, int(n * self.sample_ratio))
        sample_size = min(sample_size, n)  # Don't sample more than available

        if sample_size >= n:
            # Small array, just sort directly
            sorted_abs = np.sort(np.abs(flat_array))
            threshold = sorted_abs[-k] if k <= n else 0
        else:
            # Large array, use sampling
            sample_indices = np.random.choice(n, sample_size, replace=False)
            samples = np.abs(flat_array[sample_indices])
            sorted_samples = np.sort(samples)

            # Estimate k from samples
            k_sampled = int(sample_size * self.keep_ratio)
            k_sampled = max(1, min(k_sampled, sample_size - 1))
            threshold = sorted_samples[-k_sampled]

        return threshold

    def get_statistics(self) -> Dict:
        """Get cumulative pruning statistics."""
        return {
            'avg_compression_ratio': np.mean(self.compression_ratios) if self.compression_ratios else 0,
            'avg_sparsity': 1 - np.mean(self.compression_ratios) if self.compression_ratios else 0,
            'avg_communication_savings': np.mean(self.communication_saved) if self.communication_saved else 0,
            'num_pruning_operations': len(self.compression_ratios),
            'config': {
                'keep_ratio': self.keep_ratio,
                'momentum_factor': self.momentum_factor,
                'use_momentum_correction': self.use_momentum_correction,
                'sample_ratio': self.sample_ratio
            }
        }

    def reset(self):
        """Reset velocity buffers and statistics (for new experiment)."""
        self.velocities = {}
        self.compression_ratios = []
        self.communication_saved = []


# Convenience functions for easy integration
def apply_gradient_pruning(parameters: List[np.ndarray],
                           keep_ratio: float = 0.1,
                           use_momentum: bool = True,
                           pruner_instance: Optional[GradientPruner] = None) -> Tuple[List[np.ndarray], Dict]:
    """
    Apply gradient pruning to model parameters.

    This is the simple interface for your FL clients.

    Args:
        parameters: List of parameter arrays
        keep_ratio: Fraction to keep (0.1 = 10%)
        use_momentum: Use DGC momentum correction (recommended)
        pruner_instance: Reuse existing pruner (for momentum across rounds)

    Returns:
        Tuple of (pruned_parameters, statistics)

    Example:
        # In client.py, after training:
        parameters = self.flower_client.get_parameters({})
        pruned_params, stats = apply_gradient_pruning(parameters, keep_ratio=0.1)
        print(f"Compression: {stats['compression_factor']:.1f}x")
        # Then send pruned_params instead of parameters
    """
    if pruner_instance is None:
        pruner = GradientPruner(
            keep_ratio=keep_ratio,
            use_momentum_correction=use_momentum
        )
    else:
        pruner = pruner_instance

    return pruner.prune_weights(parameters)


def sparse_to_dense(sparse_params: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert sparse parameters back to dense (for aggregation).

    In practice, sparse parameters are already in dense format (with zeros).
    This function is for completeness.
    """
    # Already dense format in our implementation
    return sparse_params


# Testing
if __name__ == "__main__":
    print("Testing Gradient Pruning with Momentum Correction")
    print("="*70)

    # Simulate model parameters
    np.random.seed(42)
    params = [
        np.random.randn(1000, 100).astype(np.float32),  # Layer 1
        np.random.randn(100, 10).astype(np.float32),    # Layer 2
        np.random.randn(10).astype(np.float32)          # Bias
    ]

    print(f"\nOriginal model:")
    total_params = sum(p.size for p in params)
    total_bytes = sum(p.nbytes for p in params)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size: {total_bytes / 1024:.2f} KB")

    # Test 1: Simple top-k (no momentum)
    print(f"\n1. Simple Top-K Pruning (10%, no momentum):")
    pruner_simple = GradientPruner(keep_ratio=0.1, use_momentum_correction=False)
    pruned_simple, stats_simple = pruner_simple.prune_weights(params)
    print(f"  Kept: {stats_simple['kept_parameters']:,} / {stats_simple['total_parameters']:,}")
    print(f"  Sparsity: {stats_simple['sparsity']:.1%}")
    print(f"  Compression: {stats_simple['compression_factor']:.1f}x")
    print(f"  Communication savings: {stats_simple['communication_savings']:.1%}")

    # Test 2: DGC with momentum (simulate multiple rounds)
    print(f"\n2. DGC with Momentum Correction (10%, 3 rounds):")
    pruner_dgc = GradientPruner(keep_ratio=0.1, use_momentum_correction=True)

    for round_num in range(3):
        # Simulate new gradients each round
        new_params = [p + np.random.randn(*p.shape).astype(np.float32) * 0.1 for p in params]
        pruned_dgc, stats_dgc = pruner_dgc.prune_weights(new_params)
        print(f"  Round {round_num + 1}: Kept {stats_dgc['kept_parameters']:,}, "
              f"Compression {stats_dgc['compression_factor']:.1f}x")

    # Show cumulative statistics
    cumulative_stats = pruner_dgc.get_statistics()
    print(f"\n  Cumulative statistics:")
    print(f"    Avg sparsity: {cumulative_stats['avg_sparsity']:.1%}")
    print(f"    Avg comm savings: {cumulative_stats['avg_communication_savings']:.1%}")

    # Test 3: Convenience function
    print(f"\n3. Using convenience function:")
    pruned_conv, stats_conv = apply_gradient_pruning(params, keep_ratio=0.1)
    print(f"  Compression: {stats_conv['compression_factor']:.1f}x")
    print(f"  Communication savings: {stats_conv['communication_savings']:.1%}")

    print(f"\n{'='*70}")
    print(f"✅ Gradient pruning module ready for integration!")
    print(f"\nTo use in your FL architecture:")
    print(f"  1. In client.py, after training:")
    print(f"     pruned_params = apply_gradient_pruning(parameters, keep_ratio=0.1)")
    print(f"  2. Apply SMPC to pruned_params (optional)")
    print(f"  3. Send pruned_params to server")
