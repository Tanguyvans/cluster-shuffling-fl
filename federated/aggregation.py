"""
Aggregation methods for federated learning.
Includes standard FedAvg and Byzantine-robust methods.
"""
import numpy as np
from typing import List, Tuple, Optional
from flwr.server.strategy.aggregate import aggregate


def fedavg(weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Standard FedAvg - wrapper around Flower's aggregate function.
    
    Args:
        weights_results: List of (parameters, num_examples) tuples
    
    Returns:
        Aggregated parameters
    """
    return aggregate(weights_results)


def krum(weights_results: List[Tuple[List[np.ndarray], int]], 
         num_malicious: int = 0,
         num_to_keep: int = 1,
         output_file: str = None) -> List[np.ndarray]:
    """
    Krum/Multi-Krum aggregation - selects the most representative client(s).
    
    Args:
        weights_results: List of (parameters, num_examples) tuples
        num_malicious: Number of potentially malicious clients (f)
        num_to_keep: Number of clients to keep (1 for Krum, >1 for Multi-Krum)
    
    Returns:
        Aggregated parameters
    """
    weights_list = [w for w, _ in weights_results]
    num_clients = len(weights_list)
    
    # Flatten weights for distance computation
    flattened = []
    for client_weights in weights_list:
        flat = np.concatenate([w.flatten() for w in client_weights])
        flattened.append(flat)
    
    # Compute pairwise squared distances
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = np.linalg.norm(flattened[i] - flattened[j]) ** 2
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Compute scores (sum of squared distances to k-nearest neighbors)
    # k = n - f - 2 (from Krum paper)
    k = num_clients - num_malicious - 2
    
    if k <= 0:
        print(f"Warning: k={k} <= 0, using k={max(1, num_clients//2)}")
        k = max(1, num_clients // 2)
    
    scores = []
    for i in range(num_clients):
        # Get distances to other clients and sort
        dists = [distances[i, j] for j in range(num_clients) if i != j]
        dists.sort()
        # Sum k smallest distances
        score = sum(dists[:k])
        scores.append(score)
    
    # Select best clients
    selected_indices = np.argsort(scores)[:num_to_keep]
    
    # Log all client scores for transparency
    score_msg = f"📊 Krum scores: {[f'Client {i}: {scores[i]:.4f}' for i in range(num_clients)]}"
    print(score_msg)
    
    if num_to_keep == 1:
        # Krum: select single best client
        selected_client = selected_indices[0]
        selection_msg = f"🎯 Krum selected client {selected_client} (lowest score: {scores[selected_client]:.4f})"
        print(selection_msg)
        
        # Write to output file if provided
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"{score_msg}\n")
                f.write(f"{selection_msg}\n")
        
        return weights_list[selected_client]
    else:
        # Multi-Krum: aggregate selected clients
        selected_results = [weights_results[i] for i in selected_indices]
        selection_msg = f"🎯 Multi-Krum selected clients {selected_indices.tolist()} (scores: {[f'{scores[i]:.4f}' for i in selected_indices]})"
        print(selection_msg)
        
        # Write to output file if provided
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"{score_msg}\n")
                f.write(f"{selection_msg}\n")
        
        return aggregate(selected_results)


def trimmed_mean(weights_results: List[Tuple[List[np.ndarray], int]], 
                  trim_ratio: float = 0.2) -> List[np.ndarray]:
    """
    Coordinate-wise trimmed mean aggregation.
    
    Args:
        weights_results: List of (parameters, num_examples) tuples
        trim_ratio: Fraction to trim from each end (0.2 = 20%)
    
    Returns:
        Aggregated parameters
    """
    weights_list = [w for w, _ in weights_results]
    num_clients = len(weights_list)
    trim_num = max(1, int(trim_ratio * num_clients))
    
    if trim_num * 2 >= num_clients:
        print(f"Warning: trim too large, using trim_num=1")
        trim_num = 1
    
    aggregated = []
    num_layers = len(weights_list[0])
    
    for layer_idx in range(num_layers):
        # Stack this layer from all clients
        layer_stack = np.stack([weights_list[i][layer_idx] for i in range(num_clients)])
        
        # Sort along client dimension
        sorted_params = np.sort(layer_stack, axis=0)
        
        # Trim and average
        if num_clients > 2 * trim_num:
            trimmed = sorted_params[trim_num:-trim_num]
        else:
            trimmed = sorted_params
        
        aggregated.append(np.mean(trimmed, axis=0))
    
    return aggregated


def median(weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Coordinate-wise median aggregation.
    
    Args:
        weights_results: List of (parameters, num_examples) tuples
    
    Returns:
        Aggregated parameters
    """
    weights_list = [w for w, _ in weights_results]
    num_clients = len(weights_list)
    
    aggregated = []
    num_layers = len(weights_list[0])
    
    for layer_idx in range(num_layers):
        # Stack this layer from all clients
        layer_stack = np.stack([weights_list[i][layer_idx] for i in range(num_clients)])
        
        # Compute median
        aggregated.append(np.median(layer_stack, axis=0))
    
    return aggregated


def aggregate_with_method(weights_results: List[Tuple[List[np.ndarray], int]], 
                         method: str = 'fedavg',
                         output_file: str = None,
                         **kwargs) -> List[np.ndarray]:
    """
    Main aggregation function that selects method based on config.
    
    Args:
        weights_results: List of (parameters, num_examples) tuples
        method: Aggregation method name
        output_file: Path to output.txt file for logging
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Aggregated parameters
    """
    methods = {
        'fedavg': fedavg,
        'krum': lambda w: krum(w, num_malicious=kwargs.get('num_malicious', 0), output_file=output_file),
        'multi_krum': lambda w: krum(w, 
                                     num_malicious=kwargs.get('num_malicious', 0),
                                     num_to_keep=kwargs.get('num_to_keep', 3),
                                     output_file=output_file),
        'trimmed_mean': lambda w: trimmed_mean(w, trim_ratio=kwargs.get('trim_ratio', 0.2)),
        'median': median
    }
    
    if method not in methods:
        print(f"Unknown method {method}, using fedavg")
        method = 'fedavg'
    
    print(f"Using {method} aggregation")
    return methods[method](weights_results)