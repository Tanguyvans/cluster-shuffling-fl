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
            print(f"Warning: No global model weights for client {self.id}")
            return None

        res, metrics = self.flower_client.fit(self.global_model_weights, self.id, {})
        test_metrics = self.flower_client.evaluate(res, {'name': f'Client {self.id}'})
        
        with open(self.save_results + "output.txt", "a") as fi:
            fi.write(
                f"client {self.id}: data:{metrics['len_train']} "
                f"train: {metrics['len_train']} train: {metrics['train_loss']} {metrics['train_acc']} "
                f"val: {metrics['val_loss']} {metrics['val_acc']} "
                f"test: {test_metrics['test_loss']} {test_metrics['test_acc']}\n")
        
        # Apply SMPC only if we have connections
        if len(self.connections) > 0:
            encrypted_lists, self.list_shapes = apply_smpc(res, len(self.connections) + 1, self.type_ss, self.threshold)
            # Keep the last share for this client and send others
            self.frag_weights.append(encrypted_lists.pop())
            return encrypted_lists
        else:
            return None

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
        if not self.frag_weights:
            print(f"\n[Client {self.id}] No fragments to sum")
            return None
        
        print(f"\n[Client {self.id}] Starting to sum {len(self.frag_weights)} fragments")
        print(f"[Client {self.id}] Fragment sources: {[f'Fragment {i+1}' for i in range(len(self.frag_weights))]}")
        summed_weights = sum_shares(self.frag_weights, self.type_ss)
        print(f"[Client {self.id}] Successfully summed all fragments")
        return summed_weights

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
        # First, read the length of the data
        data_length_bytes = client_socket.recv(4)
        if not data_length_bytes:
            return
        data_length = int.from_bytes(data_length_bytes, byteorder='big')

        # Now read exactly data_length bytes
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
            client_id = message.get("id")
            weights = pickle.loads(message.get("value"))
            list_shapes = message.get("list_shapes")
            
            # Store the weights for aggregation
            if not hasattr(self, 'received_weights'):
                self.received_weights = []
            self.received_weights.append((weights, 10))  # Add weight with sample count
            
            print(f"\n[Node {self.id}] Received weights from client {client_id}")
            print(f"[Node {self.id}] Current received weights count: {len(self.received_weights)}")
            print(f"[Node {self.id}] Total clients that have contributed: {', '.join([f'c0_{i+1}' for i in range(len(self.received_weights))])}")
        else:
            print("in else")

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
        """Generate clusters of clients"""
        # Get all client IDs
        client_ids = list(self.clients.keys())
        
        # Simple clustering: create clusters of size min_clients_per_cluster
        self.clusters = []
        for i in range(0, len(client_ids), min_clients_per_cluster):
            cluster = client_ids[i:i + min_clients_per_cluster]
            if len(cluster) >= min_clients_per_cluster:
                self.clusters.append(cluster)

client_weights = []

def train_client(client_obj, metrics_tracker=None):
    if settings.get('use_clustering', False):
        # Get encrypted shares from training
        frag_weights = client_obj.train()
        if frag_weights is not None:  # Vérifier si l'entraînement a réussi
            # Send shares to other clients in the cluster
            client_obj.send_frag_clients(frag_weights)
    else:
        # Regular FL training
        weights = client_obj.train()
        if weights is not None and metrics_tracker:  # Vérifier si l'entraînement a réussi
            weights_size = sys.getsizeof(pickle.dumps(weights)) / (1024 * 1024)
            metrics_tracker.record_protocol_communication(0, weights_size, "client-node")
    
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
            # Cluster generation phase
            metrics_tracker.measure_power(round_i + 1, "cluster_generation_start")
            cluster_generation([server], [node_clients], 
                              settings.get('min_number_of_clients_in_cluster', 3),  # Utiliser .get() avec une valeur par défaut
                              1)  # We have 1 node in this case
            metrics_tracker.measure_power(round_i + 1, "cluster_generation_complete")

        with open(settings['save_results'] + "output.txt", "a") as f:
            f.write(f"### ROUND {round_i + 1} ###\n")

        # Training phase
        print(f"Node 1 : Training\n")
        
        # Training phase
        metrics_tracker.measure_power(round_i + 1, "node_1_training_start")
        threads = []
        for client in node_clients.values():
            t = threading.Thread(target=train_client, args=(client, metrics_tracker))
            t.start()
            threads.append(t)
    
        for t in threads:
            t.join()
        metrics_tracker.measure_power(round_i + 1, "node_1_training_complete")

        if settings.get('use_clustering', False):
            print(f"\nNode 1 : SMPC Phase")
            
            # SMPC phase
            metrics_tracker.measure_power(round_i + 1, "node_1_smpc_start")
            for cluster_idx, cluster in enumerate(server.clusters):
                print(f"\n=== Processing Cluster {cluster_idx + 1} ===")
                print(f"Cluster members: {', '.join(cluster)}")
                cluster_weights = []  # Store weights for this cluster
                
                for client_id in cluster:
                    if client_id in ['tot', 'count']:
                        continue
                    client = node_clients[client_id]
                    print(f"\n[Cluster {cluster_idx + 1}] Processing client {client_id}")
                    print(f"[Cluster {cluster_idx + 1}] Sending fragments from {client_id} to node")
                    client.send_frag_node()
                    
                    # After sending fragments, get the summed weights
                    if client.sum_weights is not None:
                        cluster_weights.append(client.sum_weights)
                        print(f"[Cluster {cluster_idx + 1}] Successfully added summed weights from {client_id}")
                    else:
                        print(f"[Cluster {cluster_idx + 1}] No summed weights available from {client_id}")
                    time.sleep(5)

                # If we have weights from this cluster, add them to client_weights
                if cluster_weights:
                    print(f"\n[Cluster {cluster_idx + 1}] Adding {len(cluster_weights)} weights to global weights")
                    print(f"[Cluster {cluster_idx + 1}] Contributing clients: {', '.join(cluster)}")
                    client_weights.extend(cluster_weights)
                else:
                    print(f"\n[Cluster {cluster_idx + 1}] No weights to add from this cluster")

                time.sleep(settings['ts'])

        time.sleep(settings['ts'])

        # Aggregation phase
        metrics_tracker.measure_power(round_i + 1, "aggregation_start")
        server.create_global_model(client_weights, round_i + 1)
        metrics_tracker.measure_power(round_i + 1, "aggregation_complete")

        time.sleep(settings['ts'])
        client_weights = []  # Reset client weights for next round

    # Final operations
    metrics_tracker.measure_global_power("complete")
    metrics_tracker.save_metrics()
    print("\nTraining completed. Exiting program...")
