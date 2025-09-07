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
         output_file: str = None,
         client_ids: List[str] = None) -> List[np.ndarray]:
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
    for i, client_weights in enumerate(weights_list):
        flat = np.concatenate([w.flatten() for w in client_weights])
        flattened.append(flat)
        client_label = client_ids[i] if client_ids and i < len(client_ids) else f"Client_{i}"
        norm = np.linalg.norm(flat)
        debug_msg = f"DEBUG: {client_label} weight norm: {norm:.4f}, first 5 values: {flat[:5]}"
        print(debug_msg)
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"{debug_msg}\n")
    
    # Compute pairwise squared distances (as per original Krum paper)
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
    debug_msg = f"DEBUG: Krum using k={k} closest neighbors for scoring"
    print(debug_msg)
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{debug_msg}\n")
    
    for i in range(num_clients):
        # Get distances to other clients and sort
        dists = [distances[i, j] for j in range(num_clients) if i != j]
        dists.sort()
        # Sum k smallest distances
        score = sum(dists[:k])
        scores.append(score)
        
        client_label = client_ids[i] if client_ids and i < len(client_ids) else f"Client_{i}"
        debug_msg = f"DEBUG: {client_label} distances to others: {[f'{d:.2f}' for d in dists[:3]]} -> score: {score:.4f}"
        print(debug_msg)
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"{debug_msg}\n")
    
    # Select best clients
    selected_indices = np.argsort(scores)[:num_to_keep]
    
    # Log all client scores for transparency with client IDs
    if client_ids:
        score_labels = []
        for i in range(num_clients):
            client_label = client_ids[i] if i < len(client_ids) else f"Model_{i}"
            score_labels.append(f"{client_label}: {scores[i]:.4f}")
        score_msg = f"📊 Krum scores: {score_labels}"
    else:
        score_msg = f"📊 Krum scores: {[f'Client {i}: {scores[i]:.4f}' for i in range(num_clients)]}"
    print(score_msg)
    
    if num_to_keep == 1:
        # Krum: select single best client
        selected_client = selected_indices[0]
        client_label = client_ids[selected_client] if client_ids and selected_client < len(client_ids) else f"Model_{selected_client}"
        selection_msg = f"🎯 Krum selected {client_label} (lowest score: {scores[selected_client]:.4f})"
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
        if client_ids:
            selected_labels = [client_ids[i] if i < len(client_ids) else f"Model_{i}" for i in selected_indices]
            selection_msg = f"🎯 Multi-Krum selected {selected_labels} (scores: {[f'{scores[i]:.4f}' for i in selected_indices]})"
        else:
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


def fltrust(weights_results: List[Tuple[List[np.ndarray], int]], 
             server_weights: List[np.ndarray],
             previous_weights: List[np.ndarray],
             learning_rate: float = 0.01,
             output_file: str = None,
             client_ids: List[str] = None) -> List[np.ndarray]:
    """
    FLTrust aggregation - Byzantine-robust aggregation using server model trust scores.
    
    Args:
        weights_results: List of (parameters, num_examples) tuples from clients
        server_weights: Current server model parameters (from root dataset)
        previous_weights: Previous round global model parameters
        learning_rate: Learning rate for computing updates
        output_file: Path to output.txt file for logging
    
    Returns:
        Aggregated parameters
    """
    if not weights_results:
        return server_weights
        
    client_weights = [w for w, _ in weights_results]
    num_clients = len(client_weights)
    
    # Compute server update (server_weights - previous_weights)
    server_update = []
    for i in range(len(server_weights)):
        server_update.append(server_weights[i] - previous_weights[i])
    
    # Compute client updates
    client_updates = []
    for client_w in client_weights:
        update = []
        for i in range(len(client_w)):
            update.append(client_w[i] - previous_weights[i])
        client_updates.append(update)
    
    # Flatten updates for cosine similarity computation
    def flatten_params(params):
        return np.concatenate([p.flatten() for p in params])
    
    server_update_flat = flatten_params(server_update)
    client_updates_flat = [flatten_params(update) for update in client_updates]
    
    # Compute trust scores using ReLU(cosine_similarity)
    trust_scores = []
    
    # Debug output with client ID mapping
    print(f"DEBUG: Server update norm: {np.linalg.norm(server_update_flat):.6f}")
    if client_ids:
        print(f"DEBUG: Client order mapping: {[f'Index {i} -> {client_ids[i]}' for i in range(len(client_ids))]}")
    
    for i, client_update_flat in enumerate(client_updates_flat):
        # Cosine similarity
        dot_product = np.dot(server_update_flat, client_update_flat)
        server_norm = np.linalg.norm(server_update_flat)
        client_norm = np.linalg.norm(client_update_flat)
        
        if server_norm == 0 or client_norm == 0:
            # Handle zero norm case
            cos_sim = 0
        else:
            cos_sim = dot_product / (server_norm * client_norm)
        
        # Apply ReLU to exclude negatively correlated updates
        relu_cos_sim = max(0, cos_sim)
        
        # Apply FLTrust normalization (like in the GitHub repo)
        if client_norm > 0:
            normalization_factor = server_norm / client_norm
            trust_score = relu_cos_sim * normalization_factor
        else:
            trust_score = 0
            
        trust_scores.append(trust_score)
        
        client_label = client_ids[i] if client_ids and i < len(client_ids) else f"Index_{i}"
        print(f"DEBUG: Client {client_label} - norm:{client_norm:.4f}, cos_sim:{cos_sim:.4f}, norm_factor:{server_norm/client_norm if client_norm > 0 else 0:.4f}, trust:{trust_score:.4f}")
    
    # Log trust scores with client IDs
    if client_ids:
        score_msg = f"🔒 FLTrust scores: {[f'{client_ids[i]}: {trust_scores[i]:.4f}' for i in range(num_clients)]}"
    else:
        score_msg = f"🔒 FLTrust scores: {[f'Client {i}: {trust_scores[i]:.4f}' for i in range(num_clients)]}"
    print(score_msg)
    print(f"DEBUG: Server norm={np.linalg.norm(server_update_flat):.4f}, Client norms={[f'{np.linalg.norm(cu):.4f}' for cu in client_updates_flat]}")
    
    # Show which clients are included/excluded based on trust scores
    included_clients = []
    excluded_clients = []
    for i in range(num_clients):
        client_label = client_ids[i] if client_ids and i < len(client_ids) else f"Index_{i}"
        if trust_scores[i] > 0:
            included_clients.append(f"{client_label}({trust_scores[i]:.4f})")
        else:
            excluded_clients.append(f"{client_label}({trust_scores[i]:.4f})")
    
    print(f"✅ INCLUDED clients ({len(included_clients)}): {', '.join(included_clients) if included_clients else 'NONE'}")
    print(f"❌ EXCLUDED clients ({len(excluded_clients)}): {', '.join(excluded_clients) if excluded_clients else 'NONE'}")
    
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{score_msg}\n")
    
    # Check if all trust scores are zero
    total_trust = sum(trust_scores)
    if total_trust == 0:
        print("⚠️  All trust scores are zero, using server model")
        if output_file:
            with open(output_file, "a") as f:
                f.write("Warning: All trust scores are zero, using server model\n")
        return server_weights
    
    # Normalize client updates to have same magnitude as server update
    server_magnitude = np.linalg.norm(server_update_flat)
    normalized_updates = []
    
    for i, (client_update, trust_score) in enumerate(zip(client_updates, trust_scores)):
        if trust_score > 0 and server_magnitude > 0:
            client_update_flat = flatten_params(client_update)
            client_magnitude = np.linalg.norm(client_update_flat)
            
            if client_magnitude > 0:
                # Scale client update to match server magnitude
                scale_factor = server_magnitude / client_magnitude
                normalized_update = []
                for layer in client_update:
                    normalized_update.append(layer * scale_factor)
                normalized_updates.append(normalized_update)
            else:
                # Zero update case
                normalized_updates.append(client_update)
        else:
            # Zero trust score - exclude this update
            normalized_updates.append([np.zeros_like(layer) for layer in client_update])
    
    # Compute weighted average of normalized updates
    aggregated_update = []
    for layer_idx in range(len(server_update)):
        weighted_layer = np.zeros_like(server_update[layer_idx])
        
        for client_idx in range(num_clients):
            if trust_scores[client_idx] > 0:
                # Ensure proper numpy operations
                client_layer = np.array(normalized_updates[client_idx][layer_idx])
                weighted_contribution = trust_scores[client_idx] * client_layer
                weighted_layer = np.add(weighted_layer, weighted_contribution)
        
        # Normalize by total trust
        if total_trust > 0:
            weighted_layer = np.divide(weighted_layer, total_trust)
            
        aggregated_update.append(weighted_layer)
    
    # Apply aggregated update to previous weights
    aggregated_weights = []
    for i in range(len(previous_weights)):
        # Ensure proper numpy operations
        prev_weight = np.array(previous_weights[i])
        update = np.array(aggregated_update[i])
        new_weight = np.add(prev_weight, update)
        aggregated_weights.append(new_weight)
    
    # Log aggregation info
    selection_msg = f"🎯 FLTrust aggregation completed: {len(included_clients)}/{num_clients} clients included (total trust: {total_trust:.4f})"
    print(selection_msg)
    
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{selection_msg}\n")
    
    return aggregated_weights


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
        'krum': lambda w: krum(w, num_malicious=kwargs.get('num_malicious', 0), output_file=output_file, client_ids=kwargs.get('client_ids')),
        'multi_krum': lambda w: krum(w, 
                                     num_malicious=kwargs.get('num_malicious', 0),
                                     num_to_keep=kwargs.get('num_to_keep', 3),
                                     output_file=output_file,
                                     client_ids=kwargs.get('client_ids')),
        'trimmed_mean': lambda w: trimmed_mean(w, trim_ratio=kwargs.get('trim_ratio', 0.2)),
        'median': median,
        'fltrust': lambda w: fltrust(w, 
                                     server_weights=kwargs.get('server_weights'),
                                     previous_weights=kwargs.get('previous_weights'),
                                     learning_rate=kwargs.get('learning_rate', 0.01),
                                     output_file=output_file,
                                     client_ids=kwargs.get('client_ids'))
    }
    
    if method not in methods:
        print(f"Unknown method {method}, using fedavg")
        method = 'fedavg'
    
    print(f"Using {method} aggregation")
    return methods[method](weights_results)