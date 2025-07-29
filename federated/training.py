import os
import time
import socket
import pickle
import numpy as np
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

    # Save client model for this round in .npz format
    client_round_model_dir = os.path.join(settings['save_client_models'], "round_models")
    os.makedirs(client_round_model_dir, exist_ok=True)
    
    client_round_model_path = os.path.join(client_round_model_dir, f"{client_obj.id}_round_{current_round}_model.npz")
    
    try:
        # Convert weights to numpy format and save as .npz
        if isinstance(weights, list):
            # weights is a list of numpy arrays (standard format from FlowerClient)
            weights_dict = {}
            for i, weight_array in enumerate(weights):
                weights_dict[f'param_{i}'] = weight_array
        else:
            # weights might be in different format, handle accordingly
            weights_dict = {'weights': weights}
        
        with open(client_round_model_path, "wb") as fi:
            np.savez(fi, **weights_dict)
        print(f"[Client {client_obj.id}] Saved round {current_round} model to {client_round_model_path}")
        
    except Exception as e:
        print(f"[Client {client_obj.id}] Error saving round model: {e}")
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

            # Save all generated fragments/shares
            fragments_dir = os.path.join(settings['save_results'], "fragments")
            os.makedirs(fragments_dir, exist_ok=True)
            shares_to_send = list(all_shares) # Create a mutable copy for sending

            for i, share_data in enumerate(all_shares):
                # share_data is a list of numpy arrays (parameters for one share)
                share_dict = {f"param_{j}": param for j, param in enumerate(share_data)}
                if client_obj.list_shapes:
                    try:
                        share_dict["list_shapes"] = np.array(client_obj.list_shapes, dtype=object)
                    except Exception as e:
                        print(f"[Client {client_obj.id}] Warning: Could not directly save list_shapes for share {i} (round {current_round}): {e}")
                
                frag_filename = os.path.join(
                    fragments_dir,
                    f"{client_obj.id}_round_{current_round}_frag_share_{i}.npz" 
                )
                with open(frag_filename, "wb") as fi:
                    np.savez(fi, **share_dict)
                # print(f"[Client {client_obj.id}] Saved fragment share {i} to {frag_filename} (round {current_round})")
            
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
                
                # Save cluster-aggregated model
                cluster_models_dir = os.path.join(settings['save_results'], "cluster_models")
                os.makedirs(cluster_models_dir, exist_ok=True)
                cluster_model_dict = {f"param_{j}": param for j, param in enumerate(summed_w)}
                if client_obj.list_shapes: # list_shapes from the original pre-SMPC model structure
                    try:
                        cluster_model_dict["list_shapes"] = np.array(client_obj.list_shapes, dtype=object)
                    except Exception as e:
                        print(f"[Client {client_obj.id}] Warning: Could not directly save list_shapes for cluster sum (round {current_round}): {e}")

                cluster_sum_filename = os.path.join(
                    cluster_models_dir,
                    f"{client_obj.id}_round_{current_round}_cluster_sum.npz"
                )
                with open(cluster_sum_filename, "wb") as fi:
                    np.savez(fi, **cluster_model_dict)
                print(f"[Client {client_obj.id}] Saved cluster summed model to {cluster_sum_filename} (round {current_round})")

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