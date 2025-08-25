import os
import time
import socket
import pickle
import numpy as np
import torch
import sys
from security import apply_smpc
from config import settings


def train_client(client_obj, metrics_tracker=None, current_round=0, training_barrier=None):
    # Train the model - client_obj.train() should now always return raw weights or None
    print(f"[Client {client_obj.id}] Starting training for round {current_round}...")
    weights = client_obj.train() 

    if weights is None:
        print(f"[Client {client_obj.id}] Training returned no weights (round {current_round}). Not sending anything.")
        if training_barrier:
            training_barrier.wait()
        return

    print(f"[Client {client_obj.id}] Training finished for round {current_round}. Received raw weights of type: {type(weights)}")

    # Determine if we should save gradients for this round
    save_gradients = (settings.get('save_gradients', False) and 
                     current_round in settings.get('save_gradients_rounds', []))
    
    # Save client model in .pt format using the new client method
    try:
        # Prepare experiment configuration for model metadata
        experiment_config = {
            'arch': settings.get('arch', 'unknown'),
            'name_dataset': settings.get('name_dataset', 'unknown'),
            'num_classes': getattr(client_obj.flower_client, 'classes', None),
            'n_epochs': settings.get('n_epochs', 1),
            'lr': settings.get('lr', 0.001),
            'diff_privacy': settings.get('diff_privacy', False),
            'clustering': settings.get('clustering', False),
            'type_ss': settings.get('type_ss', 'additif'),
            'threshold': settings.get('threshold', 3)
        }
        
        # Fix num_classes - get length if it's a tuple/list
        if isinstance(experiment_config['num_classes'], (tuple, list)):
            experiment_config['num_classes'] = len(experiment_config['num_classes'])
        elif experiment_config['num_classes'] is None:
            experiment_config['num_classes'] = 10  # Default
            
        client_obj.save_client_model(current_round, model_weights=weights, save_gradients=save_gradients, experiment_config=experiment_config)
        
        # If gradients are requested but not yet captured, try to capture them
        if save_gradients and not hasattr(client_obj, 'last_gradients'):
            print(f"[Client {client_obj.id}] Attempting to capture gradients for attack evaluation...")
            try:
                # This would require access to the model and a sample batch
                # For now, we'll set placeholder values that can be updated by the flower client
                client_obj.last_gradients = []  # Will be populated if flower_client supports it
            except Exception as e:
                print(f"[Client {client_obj.id}] Could not capture gradients: {e}")
        
    except Exception as e:
        print(f"[Client {client_obj.id}] Error saving client model: {e}")
        import traceback
        traceback.print_exc()

    if settings.get('clustering', False):
        print(f"[Client {client_obj.id}] Clustering ENABLED for round {current_round}.")
        if not client_obj.connections:
            print(f"[Client {client_obj.id}] Clustering enabled, but client has NO connections (cluster of 1) for round {current_round}.")
        
        try:
            print(f"[Client {client_obj.id}] Applying SMPC to raw weights for {len(client_obj.connections) + 1} shares (round {current_round})...")
            # Ensure apply_smpc is called with weights which are a list of np.ndarray
            smpc_input_weights = weights # weights from client_obj.train() -> res from FlowerClient.fit
            
            all_shares, client_obj.list_shapes = apply_smpc(
                smpc_input_weights, 
                len(client_obj.connections) + 1,
                client_obj.type_ss, 
                client_obj.threshold
            )
            print(f"[Client {client_obj.id}] SMPC applied for round {current_round}. Generated {len(all_shares)} shares in total.")

            # Save all generated fragments/shares using ModelManager
            shares_to_send = list(all_shares) # Create a mutable copy for sending
            
            if client_obj.model_manager:
                for i, share_data in enumerate(all_shares):
                    # Convert share_data to PyTorch tensors
                    if isinstance(share_data, list):
                        share_dict = {}
                        for j, param in enumerate(share_data):
                            if isinstance(param, np.ndarray):
                                share_dict[f"param_{j}"] = torch.from_numpy(param)
                            elif isinstance(param, torch.Tensor):
                                share_dict[f"param_{j}"] = param.clone().detach()
                            else:
                                share_dict[f"param_{j}"] = torch.tensor(param)
                    else:
                        share_dict = {"share_data": share_data}
                    
                    fragment_metadata = {
                        'client_id': client_obj.id,
                        'share_index': i,
                        'method': client_obj.type_ss,
                        'list_shapes': client_obj.list_shapes if client_obj.list_shapes else []
                    }
                    
                    saved_paths = client_obj.model_manager.save_fragment(
                        client_id=client_obj.id,
                        round_num=current_round,
                        fragment_index=i,
                        fragment_data=share_dict,
                        fragment_metadata=fragment_metadata
                    )
                    # print(f"[Client {client_obj.id}] Saved fragment share {i} using ModelManager (round {current_round})")
            else:
                print(f"[Client {client_obj.id}] WARNING: No ModelManager available. Cannot save SMPC fragments for round {current_round}.")
            
            client_obj.frag_weights.append(shares_to_send.pop()) # Keep one share (modifies shares_to_send)
            print(f"[Client {client_obj.id}] Kept 1 share for self. Sending {len(shares_to_send)} share(s) to {len(client_obj.connections)} peer(s) (round {current_round}).")
            
            if len(shares_to_send) != len(client_obj.connections):
                print(f"[Client {client_obj.id}] WARNING (round {current_round}): Number of shares to send ({len(shares_to_send)}) does not match number of connections ({len(client_obj.connections)}). This might be an issue if not a cluster of 1.")

            if client_obj.connections: # Only send if there are actual connections
                client_obj.send_frag_clients(shares_to_send) # Send remaining shares
            
            print(f"[Client {client_obj.id}] Attempting to get summed weights (round {current_round})...")
            summed_w = client_obj.sum_weights 
            
            if summed_w is not None:
                print(f"[Client {client_obj.id}] Successfully obtained summed weights for round {current_round}. Proceeding to send to node.")
                
                # Save cluster-aggregated model using ModelManager
                if client_obj.model_manager:
                    # Convert NumPy arrays to PyTorch tensors for .pt format
                    cluster_model_state = {}
                    for j, param in enumerate(summed_w):
                        if isinstance(param, np.ndarray):
                            cluster_model_state[f"param_{j}"] = torch.from_numpy(param)
                        else:
                            cluster_model_state[f"param_{j}"] = param
                    
                    cluster_participants = list(client_obj.connections.keys()) + [client_obj.id]
                    experiment_config = {
                        'arch': settings.get('arch', 'unknown'),
                        'name_dataset': settings.get('name_dataset', 'unknown'),
                        'num_classes': 10,
                        'list_shapes': client_obj.list_shapes if client_obj.list_shapes else [],
                        'smpc_method': client_obj.type_ss,
                        'aggregation_method': 'smpc_sum'
                    }
                    
                    saved_paths = client_obj.model_manager.save_cluster_model(
                        client_id=client_obj.id,
                        cluster_id=0,  # Simple cluster ID for now
                        round_num=current_round,
                        model_state=cluster_model_state,
                        cluster_participants=cluster_participants,
                        experiment_config=experiment_config
                    )
                    print(f"[Client {client_obj.id}] Saved cluster summed model using ModelManager (round {current_round})")
                else:
                    print(f"[Client {client_obj.id}] WARNING: No ModelManager available. Cannot save cluster model for round {current_round}.")

                client_obj.send_frag_node() 
                print(f"[Client {client_obj.id}] Call to send_frag_node completed (clustering, round {current_round}).")
            else:
                print(f"[Client {client_obj.id}] Sum of fragments is None (round {current_round}). **SKIPPING send_frag_node (clustering).**")
        except Exception as e:
            print(f"[Client {client_obj.id}] Exception during clustering/SMPC process (round {current_round}): {e}")
            import traceback
            traceback.print_exc()
    else:
        # Regular FL path - send weights directly to node
        print(f"\n[Client {client_obj.id}] Clustering DISABLED (round {current_round}). Preparing to send raw weights directly to node.")
        try:
            if metrics_tracker:
                weights_size = sys.getsizeof(pickle.dumps(weights)) / (1024 * 1024)
                metrics_tracker.record_protocol_communication(current_round, weights_size, "client-node") # Use current_round
            
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            node_address = client_obj.node.get('address')
            if not node_address:
                print(f"[Client {client_obj.id}] Node address not set (round {current_round}). Cannot send weights.")
                if training_barrier:
                    training_barrier.wait()
                return

            print(f"[Client {client_obj.id}] Connecting to node at {node_address} to send raw weights (round {current_round}).")
            client_socket.connect(node_address)
            print(f"[Client {client_obj.id}] Connected to node (round {current_round}).")
            
            serialized_message = pickle.dumps({
                "type": "frag_weights", 
                "id": client_obj.id,
                "value": pickle.dumps(weights) 
            })
            
            message_length = len(serialized_message)
            print(f"[Client {client_obj.id}] Sending raw weights of size {message_length} bytes (round {current_round}).")
            
            client_socket.send(message_length.to_bytes(4, byteorder='big'))
            client_socket.send(serialized_message)
            print(f"[Client {client_obj.id}] Successfully sent raw weights to node (round {current_round}).")
            client_socket.close()
            print(f"[Client {client_obj.id}] Connection closed with node (round {current_round}).")
        except Exception as e:
            print(f"[Client {client_obj.id}] Error sending raw weights to node (non-clustering, round {current_round}): {str(e)}")

    if training_barrier:
        training_barrier.wait()