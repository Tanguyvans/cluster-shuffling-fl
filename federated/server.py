import socket
import threading
import pickle
import os
import sys
import shutil
import random
import numpy as np
import torch
import time
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
    def __init__(self, id, host, port, test, save_results, check_usefulness=True, coef_useful=1.05, tolerance_ceil=0.06, metrics_tracker=None, model_manager=None, **kwargs):
        if model_manager is None:
            raise ValueError("ModelManager is mandatory for Node. Cannot create Node without ModelManager.")
            
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
        self.model_manager = model_manager
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
        # Load from .pt files instead of .npz
        params_list = []
        for block in self.blockchain.blocks[::-1]:
            if block.model_type == "update":
                # Only support .pt format
                loaded_data = torch.load(block.storage_reference, map_location='cpu')
                if 'model_state' in loaded_data:
                    model_state = loaded_data['model_state']
                    # Same as simple_resnet_fl.py: include ALL parameters except num_batches_tracked
                    loaded_weights = [val.numpy() if isinstance(val, torch.Tensor) else val 
                                    for name, val in model_state.items() if 'num_batches_tracked' not in name]
                    len_dataset_val = loaded_data.get('len_dataset', 10)
                else:
                    # Fallback for direct weights format
                    loaded_weights = self.flower_client.get_parameters({})
                    len_dataset_val = 10
                
                loaded_weights = (loaded_weights, len_dataset_val)
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
        # Load .pt file (only supported format)
        loaded_data = torch.load(filename, map_location='cpu')
        if 'model_state' in loaded_data:
            model_state = loaded_data['model_state']
            # Same as simple_resnet_fl.py: include ALL parameters except num_batches_tracked
            loaded_weights = [val.numpy() if isinstance(val, torch.Tensor) else val 
                            for name, val in model_state.items() if 'num_batches_tracked' not in name]
        else:
            # Fallback for direct weights format
            loaded_weights = self.flower_client.get_parameters({})

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
        if not self.model_manager:
            print(f"[Node {self.id}] ERROR: No ModelManager available. Cannot create initial global model.")
            return
            
        weights_dict = self.flower_client.get_dict_params({})
        model_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                      for k, v in weights_dict.items() if k != 'len_dataset'}
        
        # Prepare experiment config - get from first client or use defaults
        experiment_config = {
            'arch': 'unknown',  # Will be filled from kwargs if available
            'name_dataset': 'unknown',
            'num_classes': 10,
            'diff_privacy': False,
            'clustering': False,
        }
        
        saved_paths = self.model_manager.save_global_model(
            node_id=self.id,
            round_num=0,
            model_state=model_state,
            contributors=[],  # Initial model has no contributors
            experiment_config=experiment_config,
            aggregation_method="initial",
            test_metrics=None
        )
        
        self.global_params_directory = saved_paths['model']
        print(f"[Node {self.id}] Saved first global model to {saved_paths['model']}")
        self.broadcast_model_to_clients(self.global_params_directory)

    def create_global_model(self, models_arg, index, two_step=False):
        if not self.model_manager:
            print(f"[Node {self.id}] ERROR: No ModelManager available. Cannot create global model for round {index}.")
            return
            
        processing_list = []  # This will hold tuples of (client_weight_params, client_num_examples, original_client_id)

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
                # Load previous model and re-save it with current round number
                previous_data = torch.load(self.global_params_directory, map_location='cpu')
                if 'model_state' in previous_data:
                    model_state = previous_data['model_state']
                else:
                    # Convert old format to new
                    weights_dict = self.flower_client.get_dict_params({})
                    model_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                                  for k, v in weights_dict.items() if k != 'len_dataset'}
                
                experiment_config = {
                    'arch': 'unknown',
                    'name_dataset': 'unknown', 
                    'num_classes': 10,
                    'diff_privacy': False,
                    'clustering': len(self.clusters) > 0 if hasattr(self, 'clusters') else False,
                }
                
                saved_paths = self.model_manager.save_global_model(
                    node_id=self.id,
                    round_num=index,
                    model_state=model_state,
                    contributors=[],
                    experiment_config=experiment_config,
                    aggregation_method="no_new_models",
                    test_metrics={}
                )
                
                self.global_params_directory = saved_paths['model']
                print(f"[Node {self.id}] Copied previous global model to round {index}")
            else: 
                print(f"[Node {self.id}] ERROR: Previous global model not found. Creating a new one for round {index}.")
                weights_dict = self.flower_client.get_dict_params({})
                model_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                              for k, v in weights_dict.items() if k != 'len_dataset'}
                
                experiment_config = {
                    'arch': 'unknown',
                    'name_dataset': 'unknown', 
                    'num_classes': 10,
                    'diff_privacy': False,
                    'clustering': len(self.clusters) > 0 if hasattr(self, 'clusters') else False,
                }
                
                saved_paths = self.model_manager.save_global_model(
                    node_id=self.id,
                    round_num=index,
                    model_state=model_state,
                    contributors=[],
                    experiment_config=experiment_config,
                    aggregation_method="initial_fallback",
                    test_metrics={}
                )
                
                self.global_params_directory = saved_paths['model']
                print(f"[Node {self.id}] Created and saved new initial global model for round {index}")

            self.broadcast_model_to_clients(self.global_params_directory)
            self.received_weights_info = [] # Clear received weights for the next round
            return

        useful_weights_for_agg = [] # List of (weight_param_object, num_examples)
        useful_client_ids_log = []
        not_useful_client_ids_log = []

        # Load current global model for comparison (PT format only)
        global_model_data = torch.load(self.global_params_directory, map_location='cpu')
        if 'model_state' in global_model_data:
            model_state = global_model_data['model_state']
            # Same as simple_resnet_fl.py: include ALL parameters except num_batches_tracked 
            global_weights_params = [val.numpy() if isinstance(val, torch.Tensor) else val 
                                   for key, val in model_state.items() 
                                   if 'num_batches_tracked' not in key and 'len_dataset' not in key]
        else:
            # Fallback: use flower client (now gets ALL parameters)
            global_weights_params = self.flower_client.get_parameters({})
        
        # Temporary directory for client models during usefulness check
        temp_model_eval_dir = os.path.join(self.save_results, "temp_client_models_for_eval")
        os.makedirs(temp_model_eval_dir, exist_ok=True)

        # For each client model received/passed
        for client_weight_params, client_num_examples, original_client_id in processing_list:
            if self.check_usefulness:
                # Sauvegarder temporairement le modèle client pour évaluation
                temp_filename = os.path.join(temp_model_eval_dir, f"temp_{original_client_id}_round_{index}.pt")
                
                self.flower_client.set_parameters(client_weight_params)
                temp_weights_dict = self.flower_client.get_dict_params({})
                
                temp_pt_data = {
                    'round': index,
                    'model_state': {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                                   for k, v in temp_weights_dict.items()},
                    'len_dataset': client_num_examples,
                    'timestamp': time.time(),
                    'client_id': original_client_id
                }
                torch.save(temp_pt_data, temp_filename)

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
            
            # Import config for aggregation method
            from config import settings
            aggregation_method = settings.get('aggregation_method', 'weights')
            
            if aggregation_method == 'gradients':
                print(f"[Node {self.id}] Using gradient-based aggregation")
                # Note: When clustering=False, we still receive model weights, not gradients
                # The gradients are saved separately for attack evaluation
                # For now, fall back to weight-based aggregation when clustering is disabled
                from config import settings
                if settings.get('clustering', False):
                    aggregated_weights = self.aggregate_gradients(useful_weights_for_agg)
                    agg_method_desc = "gradient_based_fedavg"
                else:
                    print(f"[Node {self.id}] Note: Clustering disabled, using weight aggregation (gradients saved separately)")
                    useful_weights_for_agg.append((global_weights_params, 20))
                    aggregated_weights = aggregate(useful_weights_for_agg)
                    agg_method_desc = "weights_fedavg_gradients_saved"
            else:
                print(f"[Node {self.id}] Using traditional weight-based aggregation")
                useful_weights_for_agg.append((global_weights_params, 20)) 
                print(f"[Node {self.id}] Added current global model to the aggregation set (effective models for agg: {len(useful_weights_for_agg)}).")
                aggregated_weights = aggregate(useful_weights_for_agg)
                agg_method_desc = "fedavg_with_usefulness_check"
            
            print(f"[Node {self.id}] Completed {aggregation_method}-based aggregation.")
            metrics = self.flower_client.evaluate(aggregated_weights, {})

            self.flower_client.set_parameters(aggregated_weights)
            weights_dict = self.flower_client.get_dict_params({})
            weights_dict['len_dataset'] = 0 

            # Save using ModelManager
            if self.model_manager:
                model_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                              for k, v in weights_dict.items() if k != 'len_dataset'}
                
                experiment_config = {
                    'arch': 'unknown',
                    'name_dataset': 'unknown', 
                    'num_classes': 10,
                    'diff_privacy': False,
                    'clustering': len(self.clusters) > 0 if hasattr(self, 'clusters') else False,
                }
                
                saved_paths = self.model_manager.save_global_model(
                    node_id=self.id,
                    round_num=index,
                    model_state=model_state,
                    contributors=useful_client_ids_log,
                    experiment_config=experiment_config,
                    aggregation_method=agg_method_desc,
                    test_metrics={'test_loss': metrics.get('test_loss'), 'test_acc': metrics.get('test_acc')}
                )
                
                self.global_params_directory = saved_paths['model']
                print(f"[Node {self.id}] Saved aggregated global model for round {index} to {saved_paths['model']}")
            else:
                print(f"[Node {self.id}] ERROR: No ModelManager available. Cannot save global model for round {index}.")
                return

            with open(self.save_results + "output.txt", "a") as fi:
                fi.write(f"Round {index}, Global aggregation after usefulness check: {metrics}\n")
        else:
            print(f"\n[Node {self.id}] No useful client models found for round {index}. Keeping current global model.")
            if os.path.exists(self.global_params_directory):
                # Load previous model and re-save it with current round number
                previous_data = torch.load(self.global_params_directory, map_location='cpu')
                if 'model_state' in previous_data:
                    model_state = previous_data['model_state']
                else:
                    # Convert old format to new
                    weights_dict = self.flower_client.get_dict_params({})
                    model_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                                  for k, v in weights_dict.items() if k != 'len_dataset'}
                
                experiment_config = {
                    'arch': 'unknown',
                    'name_dataset': 'unknown', 
                    'num_classes': 10,
                    'diff_privacy': False,
                    'clustering': len(self.clusters) > 0 if hasattr(self, 'clusters') else False,
                }
                
                saved_paths = self.model_manager.save_global_model(
                    node_id=self.id,
                    round_num=index,
                    model_state=model_state,
                    contributors=useful_client_ids_log,
                    experiment_config=experiment_config,
                    aggregation_method="no_useful_models",
                    test_metrics={}
                )
                
                self.global_params_directory = saved_paths['model']
                print(f"[Node {self.id}] Copied previous global model to round {index} as no useful models found.")
            else: 
                print(f"[Node {self.id}] ERROR: Previous global model not found and no useful models.")
                weights_dict = self.flower_client.get_dict_params({})
                model_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                              for k, v in weights_dict.items() if k != 'len_dataset'}
                
                experiment_config = {
                    'arch': 'unknown',
                    'name_dataset': 'unknown', 
                    'num_classes': 10,
                    'diff_privacy': False,
                    'clustering': len(self.clusters) > 0 if hasattr(self, 'clusters') else False,
                }
                
                saved_paths = self.model_manager.save_global_model(
                    node_id=self.id,
                    round_num=index,
                    model_state=model_state,
                    contributors=useful_client_ids_log,
                    experiment_config=experiment_config,
                    aggregation_method="emergency_fallback",
                    test_metrics={}
                )
                
                self.global_params_directory = saved_paths['model']
                print(f"[Node {self.id}] Created and saved new initial global model for round {index}")


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

        model_data = torch.load(model_directory, map_location='cpu')
        if 'model_state' in model_data:
            model_state = model_data['model_state']
            # Same as simple_resnet_fl.py: include ALL parameters except num_batches_tracked
            loaded_weights = [val.numpy() if isinstance(val, torch.Tensor) else val 
                            for key, val in model_state.items() 
                            if 'num_batches_tracked' not in key and 'len_dataset' not in key]
        else:
            # Fallback for direct weights format
            loaded_weights = self.flower_client.get_parameters({})
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
    
    def aggregate_gradients(self, client_gradients_list, learning_rate=0.01):
        """
        Aggregate gradients from multiple clients and apply them to the global model.
        
        Args:
            client_gradients_list: List of (gradients, num_examples, client_id) tuples
            learning_rate: Learning rate for applying gradients to global model
            
        Returns:
            Updated model parameters (same format as traditional weight aggregation)
        """
        print(f"[Node {self.id}] Starting gradient-based aggregation with {len(client_gradients_list)} clients")
        
        if not client_gradients_list:
            # Return current global model parameters if no gradients to aggregate
            return self.flower_client.get_parameters({})
        
        # Extract gradients (first element of each tuple)
        all_gradients = [gradients for gradients, _ in client_gradients_list]
        
        # Average the gradients across all clients
        averaged_gradients = []
        num_layers = len(all_gradients[0])
        
        for layer_idx in range(num_layers):
            # Stack gradients for this layer from all clients
            layer_gradients = [client_grads[layer_idx] for client_grads in all_gradients]
            # Convert to numpy if needed and stack
            if isinstance(layer_gradients[0], torch.Tensor):
                layer_gradients = [grad.cpu().numpy() for grad in layer_gradients]
            
            # Average across clients
            stacked_gradients = np.stack(layer_gradients, axis=0)
            averaged_gradient = np.mean(stacked_gradients, axis=0)
            averaged_gradients.append(averaged_gradient)
        
        # Get current global model parameters
        current_params = self.flower_client.get_parameters({})
        
        # Apply averaged gradients to current parameters
        updated_params = []
        for i, (param, avg_grad) in enumerate(zip(current_params, averaged_gradients)):
            if isinstance(param, np.ndarray) and isinstance(avg_grad, np.ndarray):
                # Check shape compatibility
                if param.shape == avg_grad.shape:
                    # Gradient descent update: param = param - learning_rate * gradient
                    updated_param = param - learning_rate * avg_grad
                else:
                    print(f"[Node {self.id}] Warning: Shape mismatch at layer {i}: param {param.shape} vs grad {avg_grad.shape}")
                    # Skip this parameter update and keep original
                    updated_param = param
            else:
                updated_param = param  # Skip non-array parameters
            updated_params.append(updated_param)
        
        print(f"[Node {self.id}] Applied averaged gradients with learning rate {learning_rate}")
        print(f"[Node {self.id}] Updated {len(updated_params)} parameter tensors")
        
        return updated_params