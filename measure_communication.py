#!/usr/bin/env python3
"""
Communication Measurement Tool

Measures communication overhead with and without gradient pruning.
Generates comparison reports showing compression benefits.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import sys


class CommunicationMeasurement:
    """Measure communication overhead in federated learning"""

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"comm_measurement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.measurements = {
            'without_pruning': [],
            'with_pruning': []
        }

    def measure_parameters_size(self, parameters: List[np.ndarray]) -> Dict:
        """Measure size of parameters in bytes and MB"""
        total_bytes = sum(p.nbytes for p in parameters)
        total_params = sum(p.size for p in parameters)

        return {
            'total_parameters': total_params,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'num_layers': len(parameters),
            'layer_sizes': [p.size for p in parameters]
        }

    def measure_sparse_parameters_size(self, parameters: List[np.ndarray]) -> Dict:
        """Measure size of sparse parameters (only non-zero values)"""
        total_nonzero = sum(np.count_nonzero(p) for p in parameters)
        total_params = sum(p.size for p in parameters)

        # For sparse transmission, we need:
        # - Values (float32)
        # - Indices (int32)
        # - Shape info (overhead)
        value_bytes = total_nonzero * 4  # float32
        index_bytes = total_nonzero * 4  # int32 for indices
        overhead_bytes = len(parameters) * 16  # shape info per layer

        total_bytes = value_bytes + index_bytes + overhead_bytes

        return {
            'total_parameters': total_params,
            'nonzero_parameters': total_nonzero,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'sparsity': 1 - (total_nonzero / total_params),
            'compression_factor': total_params / total_nonzero if total_nonzero > 0 else 0,
            'num_layers': len(parameters)
        }

    def record_communication(self, round_num: int, client_id: str,
                           parameters: List[np.ndarray],
                           is_pruned: bool = False,
                           pruning_stats: Dict = None):
        """Record communication for one client in one round"""

        if is_pruned:
            size_info = self.measure_sparse_parameters_size(parameters)
            if pruning_stats:
                size_info.update(pruning_stats)
        else:
            size_info = self.measure_parameters_size(parameters)
            size_info['sparsity'] = 0.0
            size_info['compression_factor'] = 1.0

        measurement = {
            'round': round_num,
            'client_id': client_id,
            'timestamp': datetime.now().isoformat(),
            **size_info
        }

        key = 'with_pruning' if is_pruned else 'without_pruning'
        self.measurements[key].append(measurement)

    def calculate_statistics(self, measurements: List[Dict]) -> Dict:
        """Calculate aggregate statistics from measurements"""
        if not measurements:
            return {}

        total_mb = sum(m['total_mb'] for m in measurements)
        avg_mb_per_client = total_mb / len(measurements)
        avg_compression = np.mean([m.get('compression_factor', 1.0) for m in measurements])
        avg_sparsity = np.mean([m.get('sparsity', 0.0) for m in measurements])

        return {
            'total_communication_mb': total_mb,
            'avg_communication_per_client_mb': avg_mb_per_client,
            'avg_compression_factor': avg_compression,
            'avg_sparsity': avg_sparsity,
            'num_measurements': len(measurements),
            'total_communication_gb': total_mb / 1024
        }

    def generate_comparison_report(self, output_dir: str = None):
        """Generate comparison report between pruned and unpruned communication"""

        if output_dir is None:
            output_dir = f"results/{self.experiment_name}"

        os.makedirs(output_dir, exist_ok=True)

        # Calculate statistics
        stats_without = self.calculate_statistics(self.measurements['without_pruning'])
        stats_with = self.calculate_statistics(self.measurements['with_pruning'])

        # Calculate savings
        if stats_without.get('total_communication_mb', 0) > 0:
            savings_mb = stats_without['total_communication_mb'] - stats_with.get('total_communication_mb', 0)
            savings_percent = (savings_mb / stats_without['total_communication_mb']) * 100
        else:
            savings_mb = 0
            savings_percent = 0

        # Generate text report
        report = f"""
{'=' * 80}
Communication Overhead Comparison Report
{'=' * 80}

Experiment: {self.experiment_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 80}
WITHOUT GRADIENT PRUNING (Baseline)
{'=' * 80}

Total Communication:        {stats_without.get('total_communication_mb', 0):.2f} MB ({stats_without.get('total_communication_gb', 0):.3f} GB)
Avg per Client:            {stats_without.get('avg_communication_per_client_mb', 0):.2f} MB
Number of Transmissions:   {stats_without.get('num_measurements', 0)}

{'=' * 80}
WITH GRADIENT PRUNING (DGC)
{'=' * 80}

Total Communication:        {stats_with.get('total_communication_mb', 0):.2f} MB ({stats_with.get('total_communication_gb', 0):.3f} GB)
Avg per Client:            {stats_with.get('avg_communication_per_client_mb', 0):.2f} MB
Avg Compression Factor:    {stats_with.get('avg_compression_factor', 0):.1f}x
Avg Sparsity:              {stats_with.get('avg_sparsity', 0) * 100:.1f}%
Number of Transmissions:   {stats_with.get('num_measurements', 0)}

{'=' * 80}
COMMUNICATION SAVINGS
{'=' * 80}

Absolute Savings:          {savings_mb:.2f} MB
Percentage Reduction:      {savings_percent:.1f}%

Bandwidth Reduction:       {stats_with.get('avg_compression_factor', 1):.1f}x less data transmitted

{'=' * 80}
IMPACT ANALYSIS
{'=' * 80}

"""

        # Add impact analysis
        if savings_percent > 90:
            report += "EXCELLENT: >90% communication reduction - Suitable for edge devices\n"
        elif savings_percent > 80:
            report += "VERY GOOD: 80-90% communication reduction - Significant bandwidth savings\n"
        elif savings_percent > 70:
            report += "GOOD: 70-80% communication reduction - Meaningful improvement\n"
        elif savings_percent > 50:
            report += "MODERATE: 50-70% communication reduction - Noticeable improvement\n"
        else:
            report += "LIMITED: <50% communication reduction - Consider adjusting keep_ratio\n"

        # Calculate time savings (assuming 10 Mbps network)
        network_speed_mbps = 10
        time_without = (stats_without.get('total_communication_mb', 0) * 8) / network_speed_mbps
        time_with = (stats_with.get('total_communication_mb', 0) * 8) / network_speed_mbps
        time_saved = time_without - time_with

        report += f"\nEstimated Time Savings (10 Mbps network):\n"
        report += f"  Without pruning: {time_without:.1f} seconds\n"
        report += f"  With pruning:    {time_with:.1f} seconds\n"
        report += f"  Time saved:      {time_saved:.1f} seconds ({(time_saved/time_without)*100:.1f}%)\n"

        report += f"\n{'=' * 80}\n"

        # Save text report
        report_path = os.path.join(output_dir, "communication_comparison.txt")
        with open(report_path, 'w') as f:
            f.write(report)

        print(report)
        print(f"✅ Report saved to: {report_path}")

        # Save JSON data (convert numpy types to native Python types)
        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        json_data = convert_to_native({
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'without_pruning': stats_without,
                'with_pruning': stats_with
            },
            'savings': {
                'absolute_mb': savings_mb,
                'percentage': savings_percent,
                'compression_factor': stats_with.get('avg_compression_factor', 1.0)
            },
            'measurements': self.measurements
        })

        json_path = os.path.join(output_dir, "communication_data.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"✅ JSON data saved to: {json_path}")

        return report_path, json_path


def simulate_fl_communication(num_clients: int = 6, num_rounds: int = 3,
                              model_size_mb: float = 50.0, keep_ratio: float = 0.1):
    """Simulate FL communication with and without gradient pruning"""

    print(f"\n{'=' * 80}")
    print(f"Simulating Federated Learning Communication")
    print(f"{'=' * 80}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Model size: {model_size_mb:.1f} MB")
    print(f"Keep ratio: {keep_ratio * 100:.0f}%")
    print(f"{'=' * 80}\n")

    # Create measurement tracker
    measurement = CommunicationMeasurement(
        experiment_name=f"simulation_c{num_clients}_r{num_rounds}_k{int(keep_ratio*100)}"
    )

    # Simulate model parameters (approximation)
    total_params = int((model_size_mb * 1024 * 1024) / 4)  # float32

    # Create dummy parameters (distributed across layers)
    layer_sizes = [total_params // 5, total_params // 3, total_params // 4, total_params - (total_params // 5 + total_params // 3 + total_params // 4)]

    print("Simulating communication...\n")

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}:")

        for client_id in range(num_clients):
            client_name = f"c0_{client_id}"

            # Create dummy parameters
            params = [np.random.randn(size).astype(np.float32) for size in layer_sizes]

            # Record baseline (without pruning)
            measurement.record_communication(
                round_num=round_num,
                client_id=client_name,
                parameters=params,
                is_pruned=False
            )

            # Create pruned parameters (top-k selection)
            pruned_params = []
            total_kept = 0
            total_size = 0

            for p in params:
                total_size += p.size
                k = int(p.size * keep_ratio)
                threshold = np.partition(np.abs(p.flatten()), -k)[-k]
                mask = np.abs(p) >= threshold
                pruned = p * mask
                pruned_params.append(pruned)
                total_kept += np.count_nonzero(pruned)

            # Record with pruning
            pruning_stats = {
                'kept_parameters': total_kept,
                'communication_savings': 1 - (total_kept / total_size)
            }

            measurement.record_communication(
                round_num=round_num,
                client_id=client_name,
                parameters=pruned_params,
                is_pruned=True,
                pruning_stats=pruning_stats
            )

        print(f"  ✓ Recorded {num_clients} clients")

    print(f"\n✅ Simulation complete!")

    # Generate report
    print(f"\nGenerating comparison report...")
    measurement.generate_comparison_report()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Measure FL communication overhead")
    parser.add_argument('--clients', type=int, default=6, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds')
    parser.add_argument('--model-size', type=float, default=50.0, help='Model size in MB')
    parser.add_argument('--keep-ratio', type=float, default=0.1, help='Gradient pruning keep ratio')

    args = parser.parse_args()

    simulate_fl_communication(
        num_clients=args.clients,
        num_rounds=args.rounds,
        model_size_mb=args.model_size,
        keep_ratio=args.keep_ratio
    )


if __name__ == "__main__":
    main()
