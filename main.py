import logging
import numpy as np
import threading
import os
import pickle
import time
import socket
import json
import sys
import shutil
import random

from flwr.server.strategy.aggregate import aggregate
from sklearn.model_selection import train_test_split

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from flowerclient import FlowerClient
from going_modular.security import sum_shares, data_poisoning
from going_modular.utils import initialize_parameters
from going_modular.data_setup import load_dataset
from config import settings
from going_modular.security import apply_smpc, sum_shares

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from metrics import MetricsTracker

import warnings
warnings.filterwarnings("ignore")

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

class Client:
    def __init__(self, id, host, port, train, test, save_results, **kwargs):
        self.id = id
        self.host = host
        self.port = port

        # Ajout des attributs pour le secret sharing
        self.type_ss = kwargs.get('type_ss', 'additif')
        self.threshold = kwargs.get('threshold', 3)
        self.m = kwargs.get('m', 3)
        self.list_shapes = None

        self.global_model_weights = None
        self.frag_weights = []
        self.sum_dataset_number = 0

        self.node = {}
        self.connections = {}

        self.save_results = save_results

        private_key_path = f"keys/{id}_private_key.pem"
        public_key_path = f"keys/{id}_public_key.pem"

        self.get_keys(private_key_path, public_key_path)

        x_train, y_train = train
        x_test, y_test = test

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                          stratify=None)

        self.flower_client = FlowerClient.client(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            **kwargs
        )

        self.metrics_tracker = kwargs.get('metrics_tracker')

    def start_server(self):
        # same
        start_server(self.host, self.port, self.handle_message, self.id)

    def handle_message(self, client_socket):
        data_length_bytes = client_socket.recv(4)
        if not data_length_bytes:
            return
        data_length = int.from_bytes(data_length_bytes, byteorder='big')

        data = b''
        while len(data) < data_length:
            packet = client_socket.recv(data_length - len(data))
            if not packet:
                break
            data += packet

        if len(data) < data_length:
            print("Data was truncated or connection was closed prematurely.")
            return

        message = pickle.loads(data)
        message_type = message.get("type")

        if message_type == "frag_weights":
            weights = pickle.loads(message.get("value"))
            self.frag_weights.append(weights)
        elif message_type == "global_model":
            weights = pickle.loads(message.get("value"))
            self.global_model_weights = weights
            self.flower_client.set_parameters(weights)
        elif message_type == "first_global_model":
            weights = pickle.loads(message.get("value"))
            self.global_model_weights = weights
            self.flower_client.set_parameters(weights)
            print(f"client {self.id} received the global model")

        client_socket.close()

    def train(self):
        if self.global_model_weights is None:
            print(f"[Client {self.id}] Warning: No global model weights for client {self.id}. Training cannot proceed.")
            return None # Return None if training cannot start

        res, metrics = self.flower_client.fit(self.global_model_weights, self.id, {})
        test_metrics = self.flower_client.evaluate(res, {'name': f'Client {self.id}'})
        
        with open(self.save_results + "output.txt", "a") as fi:
            fi.write(
                f"client {self.id}: data:{metrics['len_train']} "
                f"train: {metrics['len_train']} train: {metrics['train_loss']} {metrics['train_acc']} "
                f"val: {metrics['val_loss']} {metrics['val_acc']} "
                f"test: {test_metrics['test_loss']} {test_metrics['test_acc']}\n")
        
        # Always return the raw weights from training. 
        # SMPC will be handled by the train_client function if clustering is enabled.
        print(f"[Client {self.id}] Training complete. Returning raw weights to train_client function.")
        return res 

    def send_frag_clients(self, frag_weights):
        for i, (k, v) in enumerate(self.connections.items()):
            # Track protocol communication (SMPC)
            serialized_message = pickle.dumps({
                "type": "frag_weights",
                "value": pickle.dumps(frag_weights[i])
            })
            message_size = len(serialized_message) / (1024 * 1024)
            if self.metrics_tracker:
                self.metrics_tracker.record_protocol_communication(
                    0,  # round number
                    message_size,
                    "client-client"
                )

            # Send to other clients
            address = v.get('address')
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(address)  # Utiliser directement le tuple (host, port)
            client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
            client_socket.send(serialized_message)
            client_socket.close()

    def send_frag_node(self):
        if not self.frag_weights:  # Vérifier si nous avons des fragments
            print(f"Warning: No fragments to send for client {self.id}")
            return

        # Track protocol communication (send to node)
        serialized_message = pickle.dumps({
            "type": "frag_weights",
            "id": self.id,
            "value": pickle.dumps(self.sum_weights),
            "list_shapes": self.list_shapes
        })
        message_size = len(serialized_message) / (1024 * 1024)
        if self.metrics_tracker:
            self.metrics_tracker.record_protocol_communication(
                0,  # round number
                message_size,
                "client-node"
            )

        # Send to node
        address = self.node.get('address')
        if address is None:
            print(f"Warning: No node address for client {self.id}")
            return

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(address)  # Utiliser directement le tuple (host, port)
        client_socket.send(len(serialized_message).to_bytes(4, byteorder='big'))
        client_socket.send(serialized_message)
        client_socket.close()
        self.frag_weights = []

    def update_node_connection(self, id, address):
        with open(f"keys/{id}_public_key.pem", 'rb') as fi:
            public_key = serialization.load_pem_public_key(
                fi.read(),
                backend=default_backend()
            )
        self.node = {"address": ("127.0.0.1", address), "public_key": public_key}

    @property
    def sum_weights(self):
        if not self.connections and not self.frag_weights: # Truly no connections and no frags (e.g. non-cluster mode called this by mistake)
            print(f"\n[Client {self.id}] sum_weights called with no connections and no initial fragments. This should not happen in clustering.")
            return None

        # In clustering, client sends N-1 shares and keeps 1. So, it starts with 1 fragment in self.frag_weights.
        # It expects to receive len(self.connections) fragments from its peers.
        expected_total_fragments = len(self.connections) + 1 
        
        # If clustering is enabled but this client is in a "cluster of 1" (no connections)
        # then expected_total_fragments will be 1. apply_smpc would have been called with N=1.
        # self.frag_weights should contain its own (and only) share.
        if expected_total_fragments == 1 and self.frag_weights: # Cluster of 1, should have its own share
             print(f"\n[Client {self.id}] Operating in a cluster of 1. Using its own fragment.")
             # sum_shares with a single fragment should ideally return that fragment itself if SMPC was for N=1
             # or handle it as per the SMPC scheme's requirements for N=1.
        elif len(self.frag_weights) < expected_total_fragments:
            print(f"\n[Client {self.id}] Waiting for fragments. Have {len(self.frag_weights)}, expecting {expected_total_fragments} total.")
            timeout_sum = 15  # seconds to wait for fragments
            start_time_sum = time.time()
            while len(self.frag_weights) < expected_total_fragments:
                if time.time() - start_time_sum > timeout_sum:
                    print(f"\n[Client {self.id}] TIMEOUT waiting for fragments. Have {len(self.frag_weights)}, expected {expected_total_fragments}. Cannot sum.")
                    return None
                # Log periodically, not too often
                if int(time.time() - start_time_sum) % 2 == 0:
                    print(f"[Client {self.id}] Still waiting for fragments... Have {len(self.frag_weights)}/{expected_total_fragments} (Elapsed: {time.time() - start_time_sum:.1f}s)")
                time.sleep(0.2) # Short sleep

        if not self.frag_weights:
            print(f"\n[Client {self.id}] No fragments available to sum, even after waiting period (if any).")
            return None
        
        print(f"\n[Client {self.id}] Proceeding to sum. Have {len(self.frag_weights)} fragments. Expected for sum: {expected_total_fragments} (for additive) or >= {self.threshold} (for threshold-based).")

        # Check if the number of fragments is sufficient based on SMPC type and expectations.
        if self.type_ss == 'additif':
            if len(self.frag_weights) < expected_total_fragments:
                print(f"\n[Client {self.id}] Additive SS: Not all expected fragments received. Have {len(self.frag_weights)}, expected {expected_total_fragments}. Cannot sum correctly.")
                return None
        elif len(self.frag_weights) < self.threshold: # For Shamir-like schemes
                print(f"\n[Client {self.id}] Threshold SS: Not enough fragments to meet threshold. Have {len(self.frag_weights)}, need {self.threshold}.")
                return None
        
        print(f"[Client {self.id}] Sufficient fragments gathered ({len(self.frag_weights)}). Type: {self.type_ss}. Starting sum_shares.")
        
        try:
            summed_weights = sum_shares(self.frag_weights, self.type_ss)
            print(f"[Client {self.id}] sum_shares completed successfully.")
            return summed_weights
        except Exception as e:
            print(f"[Client {self.id}] Error during sum_shares: {e}")
            return None

    def get_keys(self, private_key_path, public_key_path):
        # same
        self.private_key, self.public_key = get_keys(private_key_path, public_key_path)

    def reset_connections(self):
        """Reset all client connections (called when clusters change)"""
        self.connections = {}
        self.frag_weights = []

    def add_connection(self, client_id, address):
        """Add a connection to another client in the same cluster"""
        with open(f"keys/{client_id}_public_key.pem", 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        self.connections[client_id] = {"address": ("127.0.0.1", address), "public_key": public_key}

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
            print(f"[Node] Received message length: {data_length} bytes")

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
            print(f"[Node] Received message of type: {message_type}")

            if message_type == "frag_weights":
                client_id = message.get("id")
                weights = pickle.loads(message.get("value"))
                
                # Store the weights for aggregation
                if not hasattr(self, 'received_weights'):
                    self.received_weights = []
                self.received_weights.append((weights, 10))
                
                print(f"\n[Node {self.id}] Received weights from client {client_id}")
                print(f"[Node {self.id}] Current received weights count: {len(self.received_weights)}")
                print(f"[Node {self.id}] Total clients that have contributed: {', '.join([f'c0_{i+1}' for i in range(len(self.received_weights))])}")
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

        filename = f"models/CFL/m0.npz"
        self.global_params_directory = filename
        os.makedirs("models/CFL/", exist_ok=True)
        
        # Ajouter le tracking du stockage
        if self.metrics_tracker:
            file_size = sys.getsizeof(pickle.dumps(weights_dict)) / (1024 * 1024)  # Convert to MB
            self.metrics_tracker.record_storage_communication(0, file_size, 'save')
            
        with open(filename, "wb") as fi:
            np.savez(fi, **weights_dict)

        self.broadcast_model_to_clients(filename)

    def create_global_model(self, weights, index, two_step=False):
        # If no weights provided, use received weights
        if not weights and hasattr(self, 'received_weights'):
            weights = [w for w, _ in self.received_weights]
            print(f"\n[Node {self.id}] Starting aggregation with {len(weights)} received weights")
            print(f"[Node {self.id}] Contributing clients: {', '.join([f'c0_{i+1}' for i in range(len(weights))])}")
        
        # If still no weights, return without updating
        if not weights:
            print("\nNo weights received from clients. Keeping current global model.")
            # Copy current global model for next round
            filename = f"models/CFL/m{index}.npz"
            shutil.copy2(self.global_params_directory, filename)
            self.global_params_directory = filename
            self.broadcast_model_to_clients(self.global_params_directory)
            return

        useful_weights = []
        useful_clients = []
        not_useful_clients = []

        # Charger le modèle global actuel pour comparaison
        global_weights_dict = np.load(self.global_params_directory)
        global_weights = [val for name, val in global_weights_dict.items() if 'bn' not in name and 'len_dataset' not in name]
        
        # Pour chaque modèle client
        for i, client_weights in enumerate(weights):
            client_id = f"c0_{i + 1}"  # Format du client ID basé sur la création des clients
            if self.check_usefulness:
                # Sauvegarder temporairement le modèle client pour évaluation
                temp_filename = f"models/CFL/temp_client_{i}_round_{index}.npz"
                
                self.flower_client.set_parameters(client_weights)
                temp_weights_dict = self.flower_client.get_dict_params({})
                temp_weights_dict['len_dataset'] = 0
                
                with open(temp_filename, "wb") as fi:
                    np.savez(fi, **temp_weights_dict)

                # Vérifier si le modèle est utile
                if self.is_update_useful(temp_filename, list(self.clients.keys())):
                    useful_weights.append((client_weights, 10))
                    useful_clients.append(client_id)
                else:
                    not_useful_clients.append(client_id)
                    
                # Nettoyer le fichier temporaire
                os.remove(temp_filename)
            else:
                useful_weights.append((client_weights, 10))
                useful_clients.append(client_id)

        # Afficher le résumé des modèles utiles et non utiles
        print(f"\n=== Round {index} Model Usefulness Summary ===")
        print(f"Total clients: {len(weights)}")
        print(f"Useful models ({len(useful_clients)}): {', '.join(useful_clients)}")
        print(f"Not useful models ({len(not_useful_clients)}): {', '.join(not_useful_clients)}")
        print("=======================================\n")

        # Écrire le résumé dans le fichier de sortie
        with open(self.save_results + "output.txt", "a") as fi:
            fi.write(f"\n=== Round {index} Model Usefulness Summary ===\n")
            fi.write(f"Total clients: {len(weights)}\n")
            fi.write(f"Useful models ({len(useful_clients)}): {', '.join(useful_clients)}\n")
            fi.write(f"Not useful models ({len(not_useful_clients)}): {', '.join(not_useful_clients)}\n")
            fi.write("=======================================\n")

        if len(useful_weights) > 0:
            print(f"\n[Node {self.id}] Starting final aggregation")
            print(f"[Node {self.id}] Number of useful models: {len(useful_weights)}")
            print(f"[Node {self.id}] Useful models from clients: {', '.join(useful_clients)}")
            
            # Ajouter le modèle global avec un poids plus important
            useful_weights.append((global_weights, 20))
            print(f"[Node {self.id}] Added global model to aggregation")
            
            # Agréger les modèles utiles
            aggregated_weights = aggregate(useful_weights)
            print(f"[Node {self.id}] Completed aggregation of all weights")
            metrics = self.flower_client.evaluate(aggregated_weights, {})

            self.flower_client.set_parameters(aggregated_weights)
            weights_dict = self.flower_client.get_dict_params({})
            weights_dict['len_dataset'] = 0

            filename = f"models/CFL/m{index}.npz"
            self.global_params_directory = filename
            
            if self.metrics_tracker:
                file_size = sys.getsizeof(pickle.dumps(weights_dict)) / (1024 * 1024)
                self.metrics_tracker.record_storage_communication(index, file_size, 'save')
                
            with open(filename, "wb") as fi:
                np.savez(fi, **weights_dict)

            with open(self.save_results + "output.txt", "a") as fi:
                fi.write(f"Round {index}, Global aggregation: {metrics}\n")
        else:
            print("\nNo useful models found. Keeping current global model.")
            # Copier le modèle global actuel pour le prochain round
            filename = f"models/CFL/m{index}.npz"
            shutil.copy2(self.global_params_directory, filename)
            self.global_params_directory = filename

        self.broadcast_model_to_clients(self.global_params_directory)

        # Clear received weights after aggregation
        if hasattr(self, 'received_weights'):
            self.received_weights = []

    def get_keys(self, private_key_path, public_key_path):
        # same
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

client_weights = []

def train_client(client_obj, metrics_tracker=None):
    # Train the model - client_obj.train() should now always return raw weights or None
    print(f"[Client {client_obj.id}] Starting training...")
    weights = client_obj.train() 

    if weights is None:
        print(f"[Client {client_obj.id}] Training returned no weights (possibly failed or no global model). Not sending anything.")
        training_barrier.wait()
        return

    print(f"[Client {client_obj.id}] Training finished. Received raw weights of type: {type(weights)}")

    if settings.get('use_clustering', False):
        print(f"[Client {client_obj.id}] Clustering ENABLED.")
        if not client_obj.connections:
            print(f"[Client {client_obj.id}] Clustering enabled, but client has NO connections (cluster of 1).")
        
        try:
            print(f"[Client {client_obj.id}] Applying SMPC to raw weights for {len(client_obj.connections) + 1} shares...")
            encrypted_lists, client_obj.list_shapes = apply_smpc(
                weights, 
                len(client_obj.connections) + 1,
                client_obj.type_ss, 
                client_obj.threshold
            )
            print(f"[Client {client_obj.id}] SMPC applied. Generated {len(encrypted_lists) +1} shares in total.")
            
            client_obj.frag_weights.append(encrypted_lists.pop()) # Keep one share
            print(f"[Client {client_obj.id}] Kept 1 share for self. Sending {len(encrypted_lists)} share(s) to {len(client_obj.connections)} peer(s).")
            if len(encrypted_lists) != len(client_obj.connections):
                print(f"[Client {client_obj.id}] WARNING: Number of shares to send ({len(encrypted_lists)}) does not match number of connections ({len(client_obj.connections)}). This might be an issue if not a cluster of 1.")

            if client_obj.connections: # Only send if there are actual connections
                client_obj.send_frag_clients(encrypted_lists)
            
            print(f"[Client {client_obj.id}] Attempting to get summed weights (will trigger sum_weights property)...")
            summed_w = client_obj.sum_weights 
            
            if summed_w is not None:
                print(f"[Client {client_obj.id}] Successfully obtained summed weights. Proceeding to send to node.")
                client_obj.send_frag_node() 
                print(f"[Client {client_obj.id}] Call to send_frag_node completed (clustering).")
            else:
                print(f"[Client {client_obj.id}] Sum of fragments is None (e.g. timeout or insufficient frags). **SKIPPING send_frag_node (clustering).**")
        except Exception as e:
            print(f"[Client {client_obj.id}] Exception during clustering/SMPC process: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Regular FL path - send weights directly to node
        print(f"\n[Client {client_obj.id}] Clustering DISABLED. Preparing to send raw weights directly to node.")
        try:
            if metrics_tracker:
                weights_size = sys.getsizeof(pickle.dumps(weights)) / (1024 * 1024)
                metrics_tracker.record_protocol_communication(0, weights_size, "client-node")
            
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            node_address = client_obj.node.get('address')
            if not node_address:
                print(f"[Client {client_obj.id}] Node address not set. Cannot send weights.")
                training_barrier.wait()
                return

            print(f"[Client {client_obj.id}] Connecting to node at {node_address} to send raw weights.")
            client_socket.connect(node_address)
            print(f"[Client {client_obj.id}] Connected to node.")
            
            serialized_message = pickle.dumps({
                "type": "frag_weights", 
                "id": client_obj.id,
                "value": pickle.dumps(weights) 
            })
            
            message_length = len(serialized_message)
            print(f"[Client {client_obj.id}] Sending raw weights of size {message_length} bytes.")
            
            client_socket.send(message_length.to_bytes(4, byteorder='big'))
            client_socket.send(serialized_message)
            print(f"[Client {client_obj.id}] Successfully sent raw weights to node.")
            client_socket.close()
            print(f"[Client {client_obj.id}] Connection closed with node.")
        except Exception as e:
            print(f"[Client {client_obj.id}] Error sending raw weights to node (non-clustering): {str(e)}")

    training_barrier.wait()

def create_nodes(test_sets, number_of_nodes, save_results, check_usefulness, coef_useful, tolerance_ceil, **kwargs):
    list_nodes = []
    for i in range(number_of_nodes):
        list_nodes.append(
            Node(
                id=f"n{i + 1}",
                host="127.0.0.1",
                port=8000 + i,  # Changed base port for nodes
                test=test_sets[i],
                save_results=save_results,
                check_usefulness=check_usefulness,
                coef_useful=coef_useful,
                tolerance_ceil=tolerance_ceil,
                **kwargs
            )
        )
    return list_nodes

def create_clients(train_sets, test_sets, node, number_of_clients, save_results, metrics_tracker, **kwargs):
    dict_clients = {}
    for num_client in range(number_of_clients):
        dataset_index = node * number_of_clients + num_client
        client_kwargs = {
            'id': f"c{node}_{num_client + 1}",
            'host': "127.0.0.1",
            'port': 9000 + (node * number_of_clients) + num_client,  # Changed port calculation
            'train': train_sets[dataset_index],
            'test': test_sets[dataset_index],
            'save_results': save_results,
            **kwargs
        }
        
        # Only add secret sharing parameters if they exist in settings
        if 'secret_sharing' in settings:
            client_kwargs.update({
                'type_ss': settings['secret_sharing'],
                'threshold': settings['k'],
                'm': settings['m']
            })
        
        # Create the client
        client = Client(**client_kwargs)
        # Set metrics_tracker after client creation
        client.metrics_tracker = metrics_tracker
        
        dict_clients[f"c{node}_{num_client + 1}"] = client

    return dict_clients

def cluster_generation(nodes, clients, min_number_of_clients_in_cluster, number_of_nodes):
    """Generate new clusters for each node"""
    if min_number_of_clients_in_cluster is None:
        min_number_of_clients_in_cluster = 3  # Valeur par défaut
        
    for num_node in range(number_of_nodes):
        # Reset all client connections
        for client in clients[num_node].values():
            client.reset_connections()
        
        # Generate new clusters
        nodes[num_node].generate_clusters(min_number_of_clients_in_cluster)
        
        # Set up connections within each cluster
        for cluster in nodes[num_node].clusters:
            for client_id_1 in cluster:
                if client_id_1 in ['tot', 'count']:
                    continue
                for client_id_2 in cluster:
                    if client_id_2 in ['tot', 'count'] or client_id_1 == client_id_2:
                        continue
                    clients[num_node][client_id_1].add_connection(
                        client_id_2, 
                        clients[num_node][client_id_2].port
                    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    training_barrier, length = initialize_parameters(settings, 'CFL')
    
    # Create metrics tracker at the very beginning
    metrics_tracker = MetricsTracker(settings['save_results'])
    metrics_tracker.start_tracking()
    metrics_tracker.measure_global_power("start")
    
    # Initial setup phase
    metrics_tracker.measure_power(0, "initial_setup_start")

    json_dict = {
        'settings': {**settings, "length": length}
    }
    with open(settings['save_results'] + "config.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    with open(settings['save_results'] + "output.txt", "w") as f:
        f.write("")

    client_train_sets, client_test_sets, node_test_sets, list_classes = load_dataset(
        length, settings['name_dataset'],
        settings['data_root'],
        settings['n_clients'],
        1
    )

    # Data poisoning of the clients
    metrics_tracker.measure_power(0, "data_poisoning_start")
    data_poisoning(
        client_train_sets,
        poisoning_type="targeted",
        n_classes=len(list_classes),
        poisoned_number=settings['poisoned_number'],
    )
    metrics_tracker.measure_power(0, "data_poisoning_complete")

    # Create the server entity
    metrics_tracker.measure_power(0, "server_creation_start")
    server = create_nodes(
        node_test_sets, 1, 
        save_results=settings['save_results'],
        check_usefulness=settings['check_usefulness'],
        coef_useful=settings['coef_useful'],
        tolerance_ceil=settings['tolerance_ceil'],
        dp=settings['diff_privacy'], 
        model_choice=settings['arch'],
        batch_size=settings['batch_size'],
        classes=list_classes, 
        choice_loss=settings['choice_loss'],
        choice_optimizer=settings['choice_optimizer'],
        choice_scheduler=settings['choice_scheduler'],
        save_figure=None, 
        matrix_path=settings['matrix_path'],
        roc_path=settings['roc_path'],
        pretrained=settings['pretrained'],
        save_model=settings['save_model']
    )[0]
    metrics_tracker.measure_power(0, "server_creation_complete")

    # Create clients
    metrics_tracker.measure_power(0, "clients_creation_start")
    node_clients = create_clients(
        client_train_sets, 
        client_test_sets, 
        0,  # node number
        settings['n_clients'],
        save_results=settings['save_results'],
        metrics_tracker=metrics_tracker,
        dp=settings['diff_privacy'], 
        model_choice=settings['arch'],
        batch_size=settings['batch_size'],
        epochs=settings['n_epochs'], 
        classes=list_classes,
        learning_rate=settings['lr'],
        choice_loss=settings['choice_loss'],
        choice_optimizer=settings['choice_optimizer'],
        choice_scheduler=settings['choice_scheduler'],
        step_size=settings['step_size'],
        gamma=settings['gamma'],
        save_figure=None,
        matrix_path=settings['matrix_path'],
        roc_path=settings['roc_path'],
        patience=settings['patience'],
        pretrained=settings['pretrained'],
        save_model=settings['save_model']
    )
    metrics_tracker.measure_power(0, "clients_creation_complete")

    # Setup connections
    metrics_tracker.measure_power(0, "connections_setup_start")
    for client_id, client in node_clients.items():
        client.update_node_connection(server.id, server.port)

    for client_j in node_clients.values():
        server.add_client(c_id=client_j.id, client_address=("localhost", client_j.port))
    metrics_tracker.measure_power(0, "connections_setup_complete")

    print("done with the connections")

    # Start servers
    metrics_tracker.measure_power(0, "servers_start")
    threading.Thread(target=server.start_server).start()
    for client in node_clients.values():
        threading.Thread(target=client.start_server).start()

    server.create_first_global_model()
    metrics_tracker.measure_power(0, "initial_setup_complete")
    
    time.sleep(10)

    client_weights = []

    # Training rounds
    for round_i in range(settings['n_rounds']):
        print(f"### ROUND {round_i + 1} ###")
        
        if settings.get('use_clustering', False):
            print(f"\nROUND {round_i + 1}: Clustering is ON. Generating clusters...")
            metrics_tracker.measure_power(round_i + 1, "cluster_generation_start")
            cluster_generation([server], [node_clients], 
                              settings.get('min_number_of_clients_in_cluster', 3), 
                              1)
            metrics_tracker.measure_power(round_i + 1, "cluster_generation_complete")
            # Log cluster information
            with open(settings['save_results'] + "output.txt", "a") as f:
                f.write(f"\nROUND {round_i + 1}: Clustering is ON. Clusters formed:\n")
                for i, cluster in enumerate(server.clusters):
                    f.write(f"  Node {server.id} - Cluster {i+1}: {cluster}\n")
                    for client_id_in_cluster in cluster:
                        client_obj = node_clients.get(client_id_in_cluster)
                        if client_obj:
                            f.write(f"    Client {client_id_in_cluster} connections: {list(client_obj.connections.keys())}\n")

        with open(settings['save_results'] + "output.txt", "a") as f:
            f.write(f"### ROUND {round_i + 1} ###\n")

        # Training phase
        print(f"\nROUND {round_i + 1}: Node 1 : Starting client training phase...\n")
        metrics_tracker.measure_power(round_i + 1, "node_1_training_start")
        threads = []
        for client in node_clients.values():
            t = threading.Thread(target=train_client, args=(client, metrics_tracker))
            t.start()
            threads.append(t)
    
        for t in threads:
            t.join()
        metrics_tracker.measure_power(round_i + 1, "node_1_training_complete")

        # Wait for all clients to finish and messages to be received
        # This sleep is primarily for the NODE to receive messages from clients.
        # Client-to-client fragment exchange should ideally complete within each train_client call (due to sum_weights timeout).
        wait_time_for_node = settings['ts'] 
        print(f"\nROUND {round_i + 1}: All client training threads joined. Waiting {wait_time_for_node}s for node to receive messages...")
        time.sleep(wait_time_for_node)

        # Aggregation phase
        print(f"\nROUND {round_i + 1}: Starting aggregation phase on node...")
        metrics_tracker.measure_power(round_i + 1, "aggregation_start")
        server.create_global_model(None, round_i + 1)  # Pass None to use received_weights
        metrics_tracker.measure_power(round_i + 1, "aggregation_complete")

        print("\nBroadcasting updated global model to clients...")
        time.sleep(settings['ts'] * 2)  # Wait for model distribution

    # Final operations
    metrics_tracker.measure_global_power("complete")
    metrics_tracker.save_metrics()
    print("\nTraining completed. Exiting program...")
