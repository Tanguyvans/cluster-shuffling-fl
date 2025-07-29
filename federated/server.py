import socket
import threading
import pickle
import os
import sys
import shutil
import random
import numpy as np
from flwr.server.strategy.aggregate import aggregate
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from .flower_client import FlowerClient


def get_keys(private_key_path, public_key_path):
    os.makedirs("keys/", exist_ok=True)
    if os.path.exists(private_key_path) and os.path.exists(public_key_path):
        with open(private_key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )

        with open(public_key_path, 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

    else:
        # Generate new keys
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Save keys to files
        with open(private_key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(public_key_path, 'wb') as f:
            f.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            )

    return private_key, public_key


def start_server(host, port, handle_message, num_node):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Node {num_node} listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_message, args=(client_socket,)).start()


class Node:
    def __init__(self, id, host, port, test, save_results, check_usefulness=True, coef_useful=1.05, tolerance_ceil=0.06, metrics_tracker=None, **kwargs):
        self.id = id
        self.host = host
        self.port = port

        self.clients = {}

        self.global_params_directory = ""

        self.save_results = save_results

        self.check_usefulness = check_usefulness
        self.coef_useful = coef_useful
        self.tolerance_ceil = tolerance_ceil

        self.metrics_tracker = metrics_tracker
        self.received_weights_info = [] # Initialize to store (weights, num_examples, client_id)

        private_key_path = f"keys/{id}_private_key.pem"
        public_key_path = f"keys/{id}_public_key.pem"
        self.get_keys(private_key_path, public_key_path)

        x_test, y_test = test

        self.flower_client = FlowerClient.node(
            x_test=x_test,
            y_test=y_test,
            **kwargs
        )

        self.clusters = []  # Add this line

    def start_server(self):
        start_server(self.host, self.port, self.handle_message, self.id)

    def handle_message(self, client_socket):
        try:
            # First, read the length of the data
            data_length_bytes = client_socket.recv(4)
            if not data_length_bytes:
                print("[Node] No data length received")
                return
            data_length = int.from_bytes(data_length_bytes, byteorder='big')
            # print(f"[Node] Received message length: {data_length} bytes") # Reduced verbosity

            # Now read exactly data_length bytes
            data = b''
            while len(data) < data_length:
                packet = client_socket.recv(data_length - len(data))
                if not packet:
                    print("[Node] Connection closed while receiving data")
                    break
                data += packet

            if len(data) < data_length:
                print(f"[Node] Data was truncated. Expected {data_length} bytes, got {len(data)} bytes")
                return

            message = pickle.loads(data)
            message_type = message.get("type")
            # print(f"[Node] Received message of type: {message_type}") # Reduced verbosity

            if message_type == "frag_weights":
                client_id = message.get("id") # Actual client ID from message
                weights = pickle.loads(message.get("value")) # Actual model parameters
                
                # Store as (parameters, num_examples_placeholder, actual_client_id)
                self.received_weights_info.append((weights, 10, client_id)) 
                
                print(f"\n[Node {self.id}] Received weights from client {client_id}")
                print(f"[Node {self.id}] Current received weights count: {len(self.received_weights_info)}")
                
                # Corrected logging for contributors:
                contributing_clients_actual_ids = [info[2] for info in self.received_weights_info]
                print(f"[Node {self.id}] Total clients that have contributed so far this round: {', '.join(sorted(contributing_clients_actual_ids))}")
            else:
                print(f"[Node] Unknown message type: {message_type}")

        except Exception as e:
            print(f"[Node] Error handling message: {str(e)}")
        finally:
            client_socket.close()

    def get_weights(self, len_dataset=10):
        # same
        params_list = []
        for block in self.blockchain.blocks[::-1]:
            if block.model_type == "update":
                loaded_weights_dict = np.load(block.storage_reference)
                loaded_weights = [val for name, val in loaded_weights_dict.items() if 'bn' not in name and 'len_dataset' not in name]
                loaded_weights = (loaded_weights, loaded_weights_dict[f'len_dataset'])
                params_list.append(loaded_weights)

            else:
                break

        if len(params_list) == 0:
            return None

        self.aggregated_params = aggregate(params_list)

        self.flower_client.set_parameters(self.aggregated_params)

        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = len_dataset
        return weights_dict

    def broadcast_model_to_clients(self, filename):
        loaded_weights_dict = np.load(filename)
        loaded_weights = [val for name, val in loaded_weights_dict.items() if 'bn' not in name and 'len_dataset' not in name]

        for k, v in self.clients.items():
            print("sending to client", k)
            address = v.get('address')

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('127.0.0.1', address[1]))

            serialized_data = pickle.dumps(loaded_weights)
            message = {"type": "global_model", "value": serialized_data}

            if self.metrics_tracker:
                # Utiliser record_protocol_communication au lieu de add_communication
                message_size = sys.getsizeof(serialized_data) / (1024 * 1024)  # Convert to MB
                self.metrics_tracker.record_protocol_communication(0, message_size, "node-client")

            serialized_message = pickle.dumps(message)
            client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
            client_socket.send(serialized_message)
            client_socket.close()

    def create_first_global_model(self):
        weights_dict = self.flower_client.get_dict_params({})
        weights_dict['len_dataset'] = 0

        global_model_dir = os.path.join(self.save_results, "global_models")
        os.makedirs(global_model_dir, exist_ok=True)
        filename = os.path.join(global_model_dir, f"node_{self.id}_round_0_global_model.npz")
        
        self.global_params_directory = filename
        # The following line was for a different directory structure, 
        # ensure "models/CFL/" is not strictly needed or adapt if other parts rely on it.
        # os.makedirs("models/CFL/", exist_ok=True) # Potentially remove if not needed by other logic
        
        # Ajouter le tracking du stockage
        if self.metrics_tracker:
            file_size = sys.getsizeof(pickle.dumps(weights_dict)) / (1024 * 1024)  # Convert to MB
            self.metrics_tracker.record_storage_communication(0, file_size, 'save')
            
        with open(filename, "wb") as fi:
            np.savez(fi, **weights_dict)
        print(f"[Node {self.id}] Saved first global model to {filename}")

        self.broadcast_model_to_clients(filename)

    def create_global_model(self, models_arg, index, two_step=False):
        processing_list = []  # This will hold tuples of (client_weight_params, client_num_examples, original_client_id)
        
        global_model_dir = os.path.join(self.save_results, "global_models")
        os.makedirs(global_model_dir, exist_ok=True)
        # Define filename for the new global model early
        new_global_model_filename = os.path.join(global_model_dir, f"node_{self.id}_round_{index}_global_model.npz")

        if models_arg:
            # If models are passed directly, create placeholder IDs and default num_examples
            # This path is not typically taken by the main training loop for client aggregation
            processing_list = [(model_params, 10, f"direct_model_{i}") for i, model_params in enumerate(models_arg)]
            print(f"\n[Node {self.id}] Processing {len(processing_list)} models passed directly as argument for round {index}.")
        elif self.received_weights_info:
            # Use weights received from clients
            processing_list = self.received_weights_info
            print(f"\n[Node {self.id}] Starting aggregation for round {index} with {len(processing_list)} received client models.")
            actual_client_ids = [info[2] for info in processing_list] # info is (weights, num_ex, client_id)
            print(f"[Node {self.id}] Contributing clients: {', '.join(sorted(actual_client_ids))}")
        else:
            # No models passed as argument and no models received from clients
            processing_list = []

        if not processing_list:
            print(f"\n[Node {self.id}] No client models to process for round {index}. Keeping current global model.")
            if os.path.exists(self.global_params_directory):
                shutil.copy2(self.global_params_directory, new_global_model_filename)
                self.global_params_directory = new_global_model_filename
                print(f"[Node {self.id}] Copied previous global model to {new_global_model_filename}")
            else: 
                print(f"[Node {self.id}] ERROR: Previous global model not found at {self.global_params_directory}. Creating a new one for round {index}.")
                # This fallback might create m0 again, then copy. Better to ensure it saves as new_global_model_filename.
                # For simplicity, we'll call a modified first_global_model or similar logic here to save with the correct round number.
                # Simplified fallback: create a new initial model and save it with the current round's filename.
                weights_dict = self.flower_client.get_dict_params({})
                weights_dict['len_dataset'] = 0
                with open(new_global_model_filename, "wb") as fi:
                    np.savez(fi, **weights_dict)
                self.global_params_directory = new_global_model_filename
                print(f"[Node {self.id}] Created and saved new initial global model for round {index} at {new_global_model_filename}")


            self.broadcast_model_to_clients(self.global_params_directory)
            self.received_weights_info = [] # Clear received weights for the next round
            return

        useful_weights_for_agg = [] # List of (weight_param_object, num_examples)
        useful_client_ids_log = []
        not_useful_client_ids_log = []

        # Load current global model for comparison
        global_weights_dict = np.load(self.global_params_directory)
        global_weights_params = [val for name, val in global_weights_dict.items() if 'bn' not in name and 'len_dataset' not in name]
        
        # Temporary directory for client models during usefulness check
        temp_model_eval_dir = os.path.join(self.save_results, "temp_client_models_for_eval")
        os.makedirs(temp_model_eval_dir, exist_ok=True)

        # For each client model received/passed
        for client_weight_params, client_num_examples, original_client_id in processing_list:
            if self.check_usefulness:
                # Sauvegarder temporairement le modèle client pour évaluation
                temp_filename = os.path.join(temp_model_eval_dir, f"temp_{original_client_id}_round_{index}.npz")
                
                self.flower_client.set_parameters(client_weight_params)
                temp_weights_dict = self.flower_client.get_dict_params({})
                temp_weights_dict['len_dataset'] = client_num_examples 
                
                with open(temp_filename, "wb") as fi:
                    np.savez(fi, **temp_weights_dict)

                if self.is_update_useful(temp_filename, list(self.clients.keys())): 
                    useful_weights_for_agg.append((client_weight_params, client_num_examples))
                    useful_client_ids_log.append(original_client_id)
                else:
                    not_useful_client_ids_log.append(original_client_id)
                    
                os.remove(temp_filename)
            else: 
                useful_weights_for_agg.append((client_weight_params, client_num_examples))
                useful_client_ids_log.append(original_client_id)
        
        if os.path.exists(temp_model_eval_dir): # Clean up temp directory
            shutil.rmtree(temp_model_eval_dir)

        # Afficher le résumé des modèles utiles et non utiles
        print(f"\n=== Round {index} Model Usefulness Summary ===")
        print(f"Total client models considered: {len(processing_list)}")
        print(f"Useful models ({len(useful_client_ids_log)}): {', '.join(sorted(useful_client_ids_log))}")
        print(f"Not useful models ({len(not_useful_client_ids_log)}): {', '.join(sorted(not_useful_client_ids_log))}")
        print("=======================================\n")

        # Écrire le résumé dans le fichier de sortie
        with open(self.save_results + "output.txt", "a") as fi:
            fi.write(f"\n=== Round {index} Model Usefulness Summary ===\n")
            fi.write(f"Total client models considered: {len(processing_list)}\n")
            fi.write(f"Useful models ({len(useful_client_ids_log)}): {', '.join(sorted(useful_client_ids_log))}\n")
            fi.write(f"Not useful models ({len(not_useful_client_ids_log)}): {', '.join(sorted(not_useful_client_ids_log))}\n")
            fi.write("=======================================\n")

        if len(useful_weights_for_agg) > 0:
            print(f"\n[Node {self.id}] Starting final aggregation with {len(useful_weights_for_agg)} useful client models.")
            print(f"[Node {self.id}] Useful models from clients: {', '.join(sorted(useful_client_ids_log))}")
            
            useful_weights_for_agg.append((global_weights_params, 20)) 
            print(f"[Node {self.id}] Added current global model to the aggregation set (effective models for agg: {len(useful_weights_for_agg)}).")
            
            aggregated_weights = aggregate(useful_weights_for_agg)
            print(f"[Node {self.id}] Completed aggregation of useful models and global model.")
            metrics = self.flower_client.evaluate(aggregated_weights, {})

            self.flower_client.set_parameters(aggregated_weights)
            weights_dict = self.flower_client.get_dict_params({})
            weights_dict['len_dataset'] = 0 

            self.global_params_directory = new_global_model_filename
            
            if self.metrics_tracker:
                file_size = sys.getsizeof(pickle.dumps(weights_dict)) / (1024 * 1024)
                self.metrics_tracker.record_storage_communication(index, file_size, 'save')
                
            with open(self.global_params_directory, "wb") as fi:
                np.savez(fi, **weights_dict)
            print(f"[Node {self.id}] Saved aggregated global model for round {index} to {self.global_params_directory}")

            with open(self.save_results + "output.txt", "a") as fi:
                fi.write(f"Round {index}, Global aggregation after usefulness check: {metrics}\n")
        else:
            print(f"\n[Node {self.id}] No useful client models found for round {index}. Keeping current global model.")
            if os.path.exists(self.global_params_directory):
                shutil.copy2(self.global_params_directory, new_global_model_filename)
                self.global_params_directory = new_global_model_filename
                print(f"[Node {self.id}] Copied previous global model to {new_global_model_filename} as no useful models found.")
            else: 
                print(f"[Node {self.id}] ERROR: Previous global model not found at {self.global_params_directory} and no useful models.")
                # Fallback: create a new initial model and save it with the current round's filename.
                weights_dict = self.flower_client.get_dict_params({})
                weights_dict['len_dataset'] = 0
                with open(new_global_model_filename, "wb") as fi:
                    np.savez(fi, **weights_dict)
                self.global_params_directory = new_global_model_filename
                print(f"[Node {self.id}] Created and saved new initial global model for round {index} at {new_global_model_filename}")


        self.broadcast_model_to_clients(self.global_params_directory)
        self.received_weights_info = []

    def get_keys(self, private_key_path, public_key_path):
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)

    def add_client(self, c_id, client_address):
        with open(f"keys/{c_id}_public_key.pem", 'rb') as fi:
            public_key = serialization.load_pem_public_key(
                fi.read(),
                backend=default_backend()
            )

        self.clients[c_id] = {"address": client_address, "public_key": public_key}

    def is_update_useful(self, model_directory, participants): 
        """
        Vérifie si la mise à jour du modèle est bénéfique
        """
        update_eval = self.evaluate_model(model_directory, participants, write=True)
        gm_eval = self.evaluate_model(self.global_params_directory, participants, write=False)

        print(f"node: {self.id} update_eval: {update_eval} gm_eval: {gm_eval}")

        allowed_unimprovement = min(self.tolerance_ceil, gm_eval[0] * (self.coef_useful - 1))
        if update_eval[0] <= gm_eval[0] + allowed_unimprovement:
            return True
        else: 
            return False

    def evaluate_model(self, model_directory, participants, write=True):
        """
        Évalue les performances d'un modèle
        """
        # Track storage communication (load)
        model_size = os.path.getsize(model_directory) / (1024 * 1024)
        if self.metrics_tracker:
            self.metrics_tracker.record_storage_communication(0, model_size, 'load')

        loaded_weights_dict = np.load(model_directory)
        loaded_weights = [val for name, val in loaded_weights_dict.items() if 'bn' not in name and 'len_dataset' not in name]
        test_metrics = self.flower_client.evaluate(loaded_weights, {'name': f'Node {self.id}_Clusters {participants}'})
        
        print(f"In evaluate Model (node: {self.id}) \tTest Loss: {test_metrics['test_loss']:.4f}, "
              f"\tAccuracy: {test_metrics['test_acc']:.2f}%")
              
        if write: 
            with open(self.save_results + 'output.txt', 'a') as f:
                f.write(f"node: {self.id} "
                        f"model: {model_directory} "
                        f"cluster: {participants} "
                        f"loss: {test_metrics['test_loss']} "
                        f"acc: {test_metrics['test_acc']} \n")

        return test_metrics['test_loss'], test_metrics['test_acc']

    def generate_clusters(self, min_clients_per_cluster):
        """Generate clusters of clients with shuffling"""
        # Get all client IDs
        client_ids = list(self.clients.keys())
        
        # Shuffle the client IDs randomly before forming clusters
        random.shuffle(client_ids)
        print(f"[Node {self.id}] Shuffled client IDs for new cluster generation: {client_ids}")

        self.clusters = []
        current_cluster = []
        for client_id in client_ids:
            current_cluster.append(client_id)
            if len(current_cluster) >= min_clients_per_cluster:
                # Check if adding more clients would still allow remaining clients to form a valid cluster
                remaining_clients = len(client_ids) - (client_ids.index(client_id) + 1)
                if remaining_clients < min_clients_per_cluster and remaining_clients > 0 :
                    # If not enough remaining to form another full cluster, add to current if it doesn't make it too big
                    # This logic is a bit more complex to ensure all clients are assigned if possible,
                    # and clusters are not too imbalanced.
                    # For a simpler approach, you can stick to the original slicing logic after shuffling.
                    # The provided slicing logic after shuffle is simpler:
                    pass # Original logic will be used below based on slicing the shuffled list
                else:
                    # Original logic for forming clusters with fixed sizes from the shuffled list
                    pass


        # Simpler logic for forming clusters after shuffling:
        self.clusters = []
        for i in range(0, len(client_ids), min_clients_per_cluster):
            cluster = client_ids[i:i + min_clients_per_cluster]
            # Ensure the last cluster also meets the minimum size requirement, 
            # or handle it (e.g., merge with a previous one, or allow smaller if it's the only way to include clients)
            # For now, we'll only add it if it meets the minimum size.
            if len(cluster) >= min_clients_per_cluster:
                self.clusters.append(cluster)
            elif cluster: # Some remaining clients, less than min_clients_per_cluster
                # Option: Add to the last formed cluster if it exists and won't become too large
                # Option: Discard these clients for this round if strict cluster sizes are needed
                # Option: Allow a smaller last cluster if that's acceptable
                print(f"[Node {self.id}] INFO: Remaining clients {cluster} are fewer than min_clients_per_cluster ({min_clients_per_cluster}) and will not form a separate cluster this round with current logic.")
                # To include them, you might append to the last cluster or handle differently based on requirements.

        print(f"[Node {self.id}] Generated clusters: {self.clusters}")