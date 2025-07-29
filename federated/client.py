import socket
import pickle
import time
import os
from sklearn.model_selection import train_test_split
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from .flower_client import FlowerClient
from security import apply_smpc, sum_shares


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
        from .server import start_server
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
        from .server import get_keys
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