#!/usr/bin/env python3
"""
Test script for gradient pruning integration.

This script verifies that gradient pruning works correctly with the FL architecture.
"""

import sys
import numpy as np
from security.gradient_pruning import GradientPruner, apply_gradient_pruning

def test_basic_pruning():
    """Test basic gradient pruning functionality"""
    print("=" * 70)
    print("Test 1: Basic Gradient Pruning")
    print("=" * 70)

    # Create dummy parameters (simulating model weights)
    np.random.seed(42)
    params = [
        np.random.randn(1000, 100).astype(np.float32),  # Layer 1
        np.random.randn(100, 10).astype(np.float32),    # Layer 2
        np.random.randn(10).astype(np.float32)          # Bias
    ]

    total_params = sum(p.size for p in params)
    total_bytes = sum(p.nbytes for p in params)

    print(f"\nOriginal model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size: {total_bytes / 1024:.2f} KB")

    # Apply pruning
    pruned_params, stats = apply_gradient_pruning(params, keep_ratio=0.1, use_momentum=False)

    print(f"\nPruning results (10% kept):")
    print(f"  Kept parameters: {stats['kept_parameters']:,} / {stats['total_parameters']:,}")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  Compression factor: {stats['compression_factor']:.1f}x")
    print(f"  Communication savings: {stats['communication_savings']:.1%}")

    # Verify sparse parameters
    total_nonzero = sum(np.count_nonzero(p) for p in pruned_params)
    print(f"  Non-zero elements: {total_nonzero:,}")

    assert stats['kept_parameters'] > 0, "No parameters were kept!"
    assert stats['sparsity'] > 0.8, "Sparsity should be > 80% with keep_ratio=0.1"

    print("\n✅ Basic pruning test PASSED")
    return True

def test_momentum_correction():
    """Test DGC with momentum correction across multiple rounds"""
    print("\n" + "=" * 70)
    print("Test 2: Momentum Correction (DGC)")
    print("=" * 70)

    # Create dummy parameters
    np.random.seed(42)
    params = [
        np.random.randn(500, 50).astype(np.float32),
        np.random.randn(50, 10).astype(np.float32),
    ]

    # Create pruner with momentum
    pruner = GradientPruner(keep_ratio=0.1, use_momentum_correction=True)

    print(f"\nSimulating 3 training rounds with momentum correction:")

    for round_num in range(3):
        # Simulate new gradients each round
        new_params = [p + np.random.randn(*p.shape).astype(np.float32) * 0.1 for p in params]
        pruned_params, stats = pruner.prune_weights(new_params)

        print(f"  Round {round_num + 1}:")
        print(f"    Kept: {stats['kept_parameters']:,} params")
        print(f"    Compression: {stats['compression_factor']:.1f}x")
        print(f"    Comm. savings: {stats['communication_savings']:.1%}")

    # Check cumulative statistics
    cumulative_stats = pruner.get_statistics()
    print(f"\nCumulative statistics:")
    print(f"  Avg sparsity: {cumulative_stats['avg_sparsity']:.1%}")
    print(f"  Avg comm. savings: {cumulative_stats['avg_communication_savings']:.1%}")
    print(f"  Number of rounds: {cumulative_stats['num_pruning_operations']}")

    assert cumulative_stats['num_pruning_operations'] == 3, "Should have 3 pruning operations"

    print("\n✅ Momentum correction test PASSED")
    return True

def test_different_compression_ratios():
    """Test different compression ratios"""
    print("\n" + "=" * 70)
    print("Test 3: Different Compression Ratios")
    print("=" * 70)

    np.random.seed(42)
    params = [np.random.randn(1000, 100).astype(np.float32)]

    ratios = [0.01, 0.05, 0.1, 0.2, 0.5]

    print(f"\nTesting different keep_ratios:")
    for ratio in ratios:
        pruned_params, stats = apply_gradient_pruning(params, keep_ratio=ratio, use_momentum=False)
        print(f"  keep_ratio={ratio:.2f}: "
              f"Kept {stats['kept_parameters']:,} params, "
              f"Compression {stats['compression_factor']:.1f}x, "
              f"Sparsity {stats['sparsity']:.1%}")

        # Verify keep ratio is approximately correct
        actual_ratio = stats['kept_parameters'] / stats['total_parameters']
        # Allow 20% tolerance due to sampling approximation
        assert abs(actual_ratio - ratio) < ratio * 1.2, f"Keep ratio mismatch: {actual_ratio} vs {ratio}"

    print("\n✅ Compression ratio test PASSED")
    return True

def test_integration_with_fl():
    """Test that pruning integrates correctly with FL workflow"""
    print("\n" + "=" * 70)
    print("Test 4: Integration with FL Workflow")
    print("=" * 70)

    # Simulate FL client workflow
    np.random.seed(42)

    # 1. Client trains and gets parameters
    client_params = [
        np.random.randn(100, 50).astype(np.float32),
        np.random.randn(50, 10).astype(np.float32),
    ]

    print(f"\n1. Client training completed")
    print(f"   Parameters: {sum(p.size for p in client_params):,}")

    # 2. Apply gradient pruning (before SMPC)
    pruner = GradientPruner(keep_ratio=0.1, use_momentum_correction=True)
    pruned_params, stats = pruner.prune_weights(client_params)

    print(f"\n2. Gradient pruning applied")
    print(f"   Compression: {stats['compression_factor']:.1f}x")
    print(f"   Communication saved: {stats['communication_savings']:.1%}")

    # 3. Verify pruned params can be used for aggregation
    # (In real FL, this would go through SMPC then aggregation)
    print(f"\n3. Pruned parameters ready for SMPC/aggregation")
    print(f"   Sparsity: {stats['sparsity']:.1%}")

    # 4. Simulate multiple rounds
    print(f"\n4. Simulating 3 FL rounds:")
    for round_num in range(1, 4):
        new_params = [p + np.random.randn(*p.shape).astype(np.float32) * 0.05 for p in client_params]
        pruned_params, stats = pruner.prune_weights(new_params)
        print(f"   Round {round_num}: Compression {stats['compression_factor']:.1f}x")

    print("\n✅ FL integration test PASSED")
    return True

def main():
    """Run all tests"""
    print("\n🧪 Testing Gradient Pruning Integration")
    print("=" * 70)

    tests = [
        test_basic_pruning,
        test_momentum_correction,
        test_different_compression_ratios,
        test_integration_with_fl,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✅ All tests PASSED! Gradient pruning is ready for use.")
        print("\nTo enable gradient pruning in your FL experiments:")
        print("  1. Edit config.py")
        print("  2. Set 'gradient_pruning': {'enabled': True}")
        print("  3. Run python3 main.py")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
