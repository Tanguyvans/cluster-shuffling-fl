import socket
import pickle
import time
import os
import torch
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
        self.model_manager = kwargs.get('model_manager')
        if self.model_manager is None:
            raise ValueError("ModelManager is mandatory for Client. Cannot create Client without ModelManager.")

        private_key_path = f"keys/{id}_private_key.pem"
        public_key_path = f"keys/{id}_public_key.pem"

        self.get_keys(private_key_path, public_key_path)

        x_train, y_train = train
        x_test, y_test = test

        # Handle single-sample training (for gradient inversion attacks)
        # If only 1 sample, use same sample for train and val (like train_ffhq_resnet.py)
        if len(x_train) == 1:
            x_val, y_val = x_train, y_train
        else:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                              stratify=None)

        # Filter out Client-specific kwargs before passing to FlowerClient
        flower_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['model_manager', 'metrics_tracker', 'poisoning_config']}

        self.flower_client = FlowerClient.client(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            **flower_kwargs
        )

        self.metrics_tracker = kwargs.get('metrics_tracker')
        
        # Store training data for gradient saving
        self.last_batch_images = None
        self.last_batch_labels = None
        self.last_batch_indices = None
        self.last_gradients = None
        self.last_loss = None
        self.last_accuracy = None
        self.last_grad_norm = None
        
        # Attack performance tracking
        self.attack_impact_log = {
            'round_losses': [],
            'round_accuracies': [], 
            'gradient_norms': [],
            'attack_effectiveness_history': [],
            'model_convergence_indicators': []
        }
        
        # Initialize attack functionality
        self.poisoning_config = kwargs.get('poisoning_config', {})
        self.is_malicious = self._check_if_malicious()
        self.attack_strategy = None
        
        if self.is_malicious:
            self._initialize_attack()

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

    def train(self, config=None):
        if self.global_model_weights is None:
            print(f"[Client {self.id}] Warning: No global model weights for client {self.id}. Training cannot proceed.")
            return None # Return None if training cannot start

        if config is None:
            config = {}
            
        # Apply data poisoning if this is a malicious client
        if self.is_malicious and self.attack_strategy:
            round_num = config.get('round_number', 0)
            self._current_round = round_num  # Store for logging
            self.attack_strategy.set_round(round_num)
            
            if self.attack_strategy.should_attack(round_num):
                print(f"[Client {self.id}] Applying {self.attack_strategy.__class__.__name__} attack in round {round_num}")
                self._poison_training_data()
            else:
                print(f"[Client {self.id}] Skipping attack in round {round_num}")
        
        res, metrics = self.flower_client.fit(self.global_model_weights, self.id, config)
        
        # Apply gradient poisoning if applicable
        if self.is_malicious and self.attack_strategy and hasattr(res, '__iter__'):
            round_info = {
                'client_id': self.id,
                'round_number': config.get('round_number', 0),
                'metrics': metrics
            }
            res = self._poison_gradients(res, round_info)
        
        test_metrics = self.flower_client.evaluate(res, {'name': f'Client {self.id}'})
        
        with open(self.save_results + "output.txt", "a") as fi:
            fi.write(
                f"client {self.id}: data:{metrics['len_train']} "
                f"train: {metrics['len_train']} train: {metrics['train_loss']} {metrics['train_acc']} "
                f"val: {metrics['val_loss']} {metrics['val_acc']} "
                f"test: {test_metrics['test_loss']} {test_metrics['test_acc']}\n")
        
        # Store comprehensive training metrics for ModelManager
        self.last_train_loss = metrics.get('train_loss', 0.0)
        self.last_train_acc = metrics.get('train_acc', 0.0)
        self.last_val_loss = metrics.get('val_loss', None)
        self.last_val_acc = metrics.get('val_acc', None)
        self.last_loss = test_metrics['test_loss']
        self.last_accuracy = test_metrics['test_acc']
        
        # Always return the raw weights from training. 
        # SMPC will be handled by the train_client function if clustering is enabled.
        print(f"[Client {self.id}] Training complete. Returning {'poisoned' if self.is_malicious else 'clean'} weights.")
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

    def save_client_model(self, round_num, model_weights=None, save_gradients=False, experiment_config=None):
        """
        Save client model and optionally gradients using ModelManager
        
        Args:
            round_num: Current training round number
            model_weights: Model weights to save (if None, gets from flower_client)
            save_gradients: Whether to save gradients for attack evaluation
            experiment_config: Experiment configuration for metadata
        """
        if not self.model_manager:
            print(f"[Client {self.id}] ERROR: No ModelManager available. Cannot save model.")
            return
            
        if model_weights is None:
            model_weights = self.flower_client.get_parameters({})
        
        # Convert weights to state dict format
        model_state = {}
        if hasattr(self.flower_client, 'model') and hasattr(self.flower_client.model, 'state_dict'):
            # If we have access to the actual model, use its state dict
            model_state = {k: v.clone().detach().cpu() for k, v in self.flower_client.model.state_dict().items()}
        else:
            # Otherwise, convert from weights list
            for i, weight in enumerate(model_weights):
                if isinstance(weight, np.ndarray):
                    model_state[f'layer_{i}'] = torch.from_numpy(weight)
                else:
                    model_state[f'layer_{i}'] = weight
        
        # Prepare training metrics
        training_metrics = {
            'train_loss': getattr(self, 'last_train_loss', 0.0),
            'train_acc': getattr(self, 'last_train_acc', 0.0),
            'val_loss': getattr(self, 'last_val_loss', None),
            'val_acc': getattr(self, 'last_val_acc', None),
            'test_loss': getattr(self, 'last_loss', 0.0),
            'test_acc': getattr(self, 'last_accuracy', 0.0),
            'len_train': getattr(self.flower_client, 'len_train', 0)
        }
        
        # Use experiment_config from parameter or create default
        if experiment_config is None:
            experiment_config = {
                'arch': getattr(self.flower_client, 'model_choice', 'unknown'),
                'name_dataset': 'unknown',
                'num_classes': 10,
                'n_epochs': getattr(self.flower_client, 'epochs', 1),
                'lr': getattr(self.flower_client, 'learning_rate', 0.001),
                'diff_privacy': getattr(self.flower_client, 'dp', False),
                'clustering': len(self.connections) > 0,
                'type_ss': self.type_ss,
                'threshold': self.threshold
            }
        
        # Prepare gradients if needed (similar to simple_federated.py format)
        gradients = None
        gradient_metadata = None
        if save_gradients and hasattr(self.flower_client, 'last_gradients') and self.flower_client.last_gradients is not None:
            gradients = self.flower_client.last_gradients
            gradient_metadata = {
                'batch_indices': getattr(self, 'last_batch_indices', []),
                'batch_labels': getattr(self.flower_client, 'last_batch_labels', torch.empty(0)),
                'batch_images': getattr(self.flower_client, 'last_batch_images', torch.empty(0)),
                'grad_norm': getattr(self.flower_client, 'last_grad_norm', 0.0),
                'loss': getattr(self.flower_client, 'last_loss', 0.0),
                'accuracy': getattr(self.flower_client, 'last_accuracy', 0.0),
                'model_architecture': experiment_config.get('arch', 'unknown'),
                'dataset': experiment_config.get('name_dataset', 'unknown')
            }
            
            print(f"[Client {self.id}] Preparing gradients for saving: {len(gradients)} gradient tensors")
            print(f"[Client {self.id}] Gradient batch: {gradient_metadata['batch_images'].shape if gradient_metadata['batch_images'].numel() > 0 else 'empty'}")
            print(f"[Client {self.id}] Gradient loss: {gradient_metadata['loss']:.4f}, accuracy: {gradient_metadata['accuracy']:.2f}%")
        
        # Save using ModelManager
        saved_paths = self.model_manager.save_client_model(
            client_id=self.id,
            round_num=round_num,
            model_state=model_state,
            training_metrics=training_metrics,
            experiment_config=experiment_config,
            gradients=gradients,
            gradient_metadata=gradient_metadata
        )
        
        print(f"[Client {self.id}] Saved model to {saved_paths['model']}")
        if 'gradients' in saved_paths:
            print(f"[Client {self.id}] Saved gradients to {saved_paths['gradients']}")
    
    def capture_gradients_from_model(self, model, loss_fn, batch_images, batch_labels):
        """
        Capture gradients from a model for a specific batch
        
        Args:
            model: The PyTorch model
            loss_fn: Loss function
            batch_images: Input batch images
            batch_labels: Input batch labels
        """
        model.eval()
        model.zero_grad()
        
        # Forward pass
        outputs = model(batch_images)
        loss = loss_fn(outputs, batch_labels)
        
        # Get gradients
        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
        
        # Store for later saving
        self.last_gradients = [g.clone().detach() for g in gradients]
        self.last_batch_images = batch_images.clone().detach().cpu()
        self.last_batch_labels = batch_labels.clone().detach().cpu()
        self.last_loss = loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(batch_labels).sum().item()
        self.last_accuracy = 100. * correct / len(batch_labels)
        
        # Calculate gradient norm
        grad_norms = [g.norm().item() for g in gradients]
        self.last_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        
    def _check_if_malicious(self) -> bool:
        """Check if this client should be malicious based on configuration."""
        if not self.poisoning_config.get('enabled', False):
            return False
            
        malicious_clients = self.poisoning_config.get('malicious_clients', [])
        return self.id in malicious_clients
        
    def _initialize_attack(self):
        """Initialize the poisoning attack strategy."""
        try:
            from attacks.poisoning import AttackFactory
            
            attack_type = self.poisoning_config.get('attack_type', 'labelflip')
            
            # Create attack-specific configuration
            attack_config = self._create_attack_config(attack_type)
            
            # Create attack instance using factory
            self.attack_strategy = AttackFactory.create_attack(attack_type, attack_config)
            
            print(f"[Client {self.id}] Initialized as MALICIOUS client with {attack_type} attack")
            print(f"[Client {self.id}] Attack config: {attack_config}")
            
        except Exception as e:
            print(f"[Client {self.id}] ERROR: Failed to initialize attack: {e}")
            self.is_malicious = False
            self.attack_strategy = None
            
    def _create_attack_config(self, attack_type: str):
        """Create configuration for specific attack type."""
        base_config = {
            'attack_intensity': self.poisoning_config.get('attack_intensity', 0.2),
            'attack_rounds': self.poisoning_config.get('attack_rounds', None),
            'attack_frequency': self.poisoning_config.get('attack_frequency', 1.0),
            'target_class': 0,
            'num_classes': 10
        }
        
        # Add attack-specific configuration
        specific_config_key = f"{attack_type}_config"
        if specific_config_key in self.poisoning_config:
            base_config.update(self.poisoning_config[specific_config_key])
            
        return base_config
        
    def _poison_training_data(self):
        """Apply data poisoning to the training data."""
        if not hasattr(self.flower_client, 'train_loader'):
            print(f"[Client {self.id}] Warning: Cannot access train_loader for poisoning")
            return
            
        # Extract data from train_loader
        try:
            train_loader = self.flower_client.train_loader
            all_data = []
            all_labels = []
            
            for batch_data, batch_labels in train_loader:
                all_data.append(batch_data)
                all_labels.append(batch_labels)
            
            x_train = torch.cat(all_data, dim=0)
            y_train = torch.cat(all_labels, dim=0)
            
            print(f"[Client {self.id}] Extracted {len(y_train)} samples from train_loader")
            
            # Apply poisoning
            poisoned_x, poisoned_y = self.attack_strategy.poison_data(x_train, y_train)
            
            # Show a simple example of what was flipped
            if isinstance(y_train, torch.Tensor) and isinstance(poisoned_y, torch.Tensor):
                changed_indices = (y_train != poisoned_y).nonzero(as_tuple=True)[0]
                if len(changed_indices) > 0:
                    # Show first few flipped examples
                    example_flips = []
                    for i, idx in enumerate(changed_indices[:3]):  # Show max 3 examples
                        old_label = y_train[idx].item()
                        new_label = poisoned_y[idx].item()
                        example_flips.append(f"{old_label}→{new_label}")
                    
                    print(f"[Client {self.id}] Label flipping: {', '.join(example_flips)} ({len(changed_indices)} total)")
                else:
                    attack_name = self.attack_strategy.__class__.__name__
                    if "SignFlipping" in attack_name or "Noise" in attack_name or "ALIE" in attack_name or "IPM" in attack_name:
                        print(f"[Client {self.id}] {attack_name} works on gradients, not data - no data changes expected")
                    else:
                        print(f"[Client {self.id}] No labels were flipped (attack conditions not met)")
            
            # Recreate train_loader with poisoned data
            from torch.utils.data import TensorDataset, DataLoader
            poisoned_dataset = TensorDataset(poisoned_x, poisoned_y)
            self.flower_client.train_loader = DataLoader(
                poisoned_dataset, 
                batch_size=train_loader.batch_size,
                shuffle=True
            )
            print(f"[Client {self.id}] ✅ Updated train_loader with poisoned data")
                
        except Exception as e:
            print(f"[Client {self.id}] ERROR during data poisoning: {e}")
            import traceback
            traceback.print_exc()
            
    def _poison_gradients(self, model_weights, round_info):
        """Apply gradient poisoning to model updates."""
        try:
            # Convert model weights to gradient-like dictionary
            gradients = {}
            model_state = {}
            
            if hasattr(self.flower_client, 'model') and hasattr(self.flower_client.model, 'state_dict'):
                # Use actual parameter names from model
                param_names = list(self.flower_client.model.state_dict().keys())
                for i, (param_name, weight) in enumerate(zip(param_names, model_weights)):
                    if isinstance(weight, torch.Tensor):
                        gradients[param_name] = weight
                        model_state[param_name] = weight.clone()
                    else:
                        gradients[param_name] = torch.tensor(weight)
                        model_state[param_name] = torch.tensor(weight)
            else:
                # Use generic parameter names
                for i, weight in enumerate(model_weights):
                    param_name = f'layer_{i}'
                    if isinstance(weight, torch.Tensor):
                        gradients[param_name] = weight
                        model_state[param_name] = weight.clone()
                    else:
                        gradients[param_name] = torch.tensor(weight)
                        model_state[param_name] = torch.tensor(weight)
            
            # Store original statistics for comparison
            original_stats = self._compute_gradient_stats(gradients)
            
            # Apply gradient poisoning
            poisoned_gradients = self.attack_strategy.poison_gradients(
                gradients, model_state, round_info)
            
            # Compute poisoned statistics
            poisoned_stats = self._compute_gradient_stats(poisoned_gradients)
            
            # Show before/after comparison
            self._show_gradient_transformation(original_stats, poisoned_stats)
            
            # Convert back to model weights format
            poisoned_weights = []
            for param_name in gradients.keys():
                if param_name in poisoned_gradients:
                    poisoned_weights.append(poisoned_gradients[param_name])
                else:
                    poisoned_weights.append(gradients[param_name])
                      
            return poisoned_weights
            
        except Exception as e:
            print(f"[Client {self.id}] ERROR during gradient poisoning: {e}")
            return model_weights
            
    def _compute_gradient_stats(self, gradients):
        """Compute statistics for gradient tensors."""
        stats = {}
        total_norm = 0
        total_params = 0
        raw_values = {}
        
        for param_name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                grad_flat = grad.flatten()
                stats[param_name] = {
                    'mean': grad_flat.mean().item(),
                    'std': grad_flat.std().item(),
                    'norm': torch.norm(grad_flat).item(),
                    'min': grad_flat.min().item(),
                    'max': grad_flat.max().item(),
                    'size': grad_flat.numel()
                }
                
                # Store first 20 raw values for detailed analysis
                raw_values[param_name] = grad_flat[:20].detach().cpu().numpy().tolist()
                
                total_norm += torch.norm(grad_flat).item() ** 2
                total_params += grad_flat.numel()
        
        stats['global'] = {
            'total_norm': total_norm ** 0.5,
            'total_params': total_params
        }
        
        stats['raw_values'] = raw_values
        
        return stats
        
    def _show_gradient_transformation(self, original_stats, poisoned_stats):
        """Show before/after gradient transformation."""
        attack_name = self.attack_strategy.__class__.__name__
        
        # Compare global statistics
        orig_norm = original_stats['global']['total_norm']
        pois_norm = poisoned_stats['global']['total_norm']
        norm_change = ((pois_norm - orig_norm) / orig_norm * 100) if orig_norm > 0 else 0
        
        print(f"[Client {self.id}] {attack_name} transformation:")
        print(f"  Global norm: {orig_norm:.6f} → {pois_norm:.6f} ({norm_change:+.2f}%)")
        
        # Show details for first few layers
        layer_names = [name for name in original_stats.keys() if name != 'global'][:3]
        for layer_name in layer_names:
            if layer_name in poisoned_stats:
                orig = original_stats[layer_name]
                pois = poisoned_stats[layer_name]
                
                mean_change = ((pois['mean'] - orig['mean']) / abs(orig['mean']) * 100) if abs(orig['mean']) > 1e-8 else 0
                std_change = ((pois['std'] - orig['std']) / orig['std'] * 100) if orig['std'] > 1e-8 else 0
                
                print(f"  {layer_name[:20]:20s}: mean {orig['mean']:+.6f}→{pois['mean']:+.6f} ({mean_change:+.1f}%), "
                      f"std {orig['std']:.6f}→{pois['std']:.6f} ({std_change:+.1f}%)")
        
        # Show actual value changes for first layer
        if 'raw_values' in original_stats and 'raw_values' in poisoned_stats:
            first_layer = layer_names[0] if layer_names else None
            if first_layer and first_layer in original_stats['raw_values']:
                orig_vals = original_stats['raw_values'][first_layer]
                pois_vals = poisoned_stats['raw_values'][first_layer]
                
                print(f"\n  First 10 gradient values from {first_layer[:20]}:")
                print(f"  {'Original':<12} → {'Poisoned':<12} {'Changed':<8}")
                print(f"  {'-'*45}")
                
                for i in range(min(10, len(orig_vals))):
                    orig_val = orig_vals[i]
                    pois_val = pois_vals[i]
                    opposite_val = -orig_val  # What perfect opposite would be
                    
                    # Determine change type
                    if orig_val * pois_val < 0:
                        if abs(pois_val - opposite_val) < abs(orig_val) * 0.1:
                            changed = "✓ INV"  # Close to perfect inverse
                        else:
                            changed = "✓ FLIP" # Sign flipped but not perfect inverse
                    elif abs(orig_val - pois_val) > 1e-8:
                        changed = "✓ MOD"   # Modified but same sign
                    else:
                        changed = ""        # No change
                    
                    print(f"  {orig_val:+.8f} → {pois_val:+.8f} (vs -{orig_val:+.8f}) {changed}")
        
        # Calculate attack effectiveness metrics
        effectiveness_metrics = self._calculate_attack_effectiveness(original_stats, poisoned_stats, attack_name)
        print(f"\n  Attack Effectiveness:")
        print(f"  - Gradient Direction Change: {effectiveness_metrics['direction_change']:.3f} (1.0 = perfect opposite)")
        print(f"  - Magnitude Preservation: {effectiveness_metrics['magnitude_preservation']:.3f} (1.0 = same magnitude)")
        print(f"  - Overall Attack Score: {effectiveness_metrics['attack_score']:.3f} (higher = more effective)")
        
        # Track performance impact over rounds
        self._track_attack_impact(effectiveness_metrics)
        
        # Save detailed transformation log
        self._save_transformation_log(attack_name, original_stats, poisoned_stats, effectiveness_metrics)
        
    def _calculate_attack_effectiveness(self, original_stats, poisoned_stats, attack_name):
        """Calculate comprehensive attack effectiveness metrics."""
        import torch
        
        # Calculate cosine similarity between original and poisoned gradients
        cosine_similarities = []
        magnitude_ratios = []
        
        for layer_name in original_stats.keys():
            if layer_name in ['global', 'raw_values']:
                continue
                
            if layer_name in poisoned_stats:
                orig_vals = torch.tensor(original_stats['raw_values'][layer_name][:20])
                pois_vals = torch.tensor(poisoned_stats['raw_values'][layer_name][:20])
                
                # Cosine similarity (for direction analysis)
                if torch.norm(orig_vals) > 1e-8 and torch.norm(pois_vals) > 1e-8:
                    cosine_sim = torch.dot(orig_vals, pois_vals) / (torch.norm(orig_vals) * torch.norm(pois_vals))
                    cosine_similarities.append(cosine_sim.item())
                    
                    # Magnitude ratio analysis
                    magnitude_ratios.append(torch.norm(pois_vals).item() / torch.norm(orig_vals).item())
        
        # Average metrics across all layers
        avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0
        avg_magnitude_ratio = sum(magnitude_ratios) / len(magnitude_ratios) if magnitude_ratios else 1
        
        # Direction change metric (0 = same direction, 1 = opposite, 0.5 = orthogonal)
        direction_change = (1 - avg_cosine_similarity) / 2
        
        # How well magnitude is preserved (1 = perfect preservation)
        magnitude_preservation = 1 - abs(1 - avg_magnitude_ratio)
        
        # Overall attack effectiveness score
        if attack_name.lower() in ['ipmattack', 'signflippingattack']:
            # For inversion attacks, we want opposite direction with similar magnitude
            attack_score = direction_change * magnitude_preservation
        elif attack_name.lower() == 'noiseattack':
            # For noise attacks, we want to disrupt both direction and magnitude
            attack_score = direction_change * (1 - magnitude_preservation)
        else:
            # General disruption metric
            attack_score = direction_change
        
        return {
            'cosine_similarity': avg_cosine_similarity,
            'direction_change': direction_change,
            'magnitude_ratio': avg_magnitude_ratio,
            'magnitude_preservation': magnitude_preservation,
            'attack_score': attack_score,
            'layers_analyzed': len(cosine_similarities)
        }
    
    def _track_attack_impact(self, effectiveness_metrics):
        """Track attack impact on model performance over rounds."""
        current_round = getattr(self, '_current_round', 0)
        
        # Store effectiveness metrics
        self.attack_impact_log['attack_effectiveness_history'].append({
            'round': current_round,
            'metrics': effectiveness_metrics.copy()
        })
        
        # Store current training metrics if available
        if hasattr(self, 'last_loss') and self.last_loss is not None:
            self.attack_impact_log['round_losses'].append({
                'round': current_round,
                'loss': self.last_loss
            })
            
        if hasattr(self, 'last_accuracy') and self.last_accuracy is not None:
            self.attack_impact_log['round_accuracies'].append({
                'round': current_round,
                'accuracy': self.last_accuracy
            })
            
        if hasattr(self, 'last_grad_norm') and self.last_grad_norm is not None:
            self.attack_impact_log['gradient_norms'].append({
                'round': current_round,
                'grad_norm': self.last_grad_norm
            })
        
        # Calculate convergence indicators
        convergence_indicators = self._calculate_convergence_indicators()
        if convergence_indicators:
            self.attack_impact_log['model_convergence_indicators'].append({
                'round': current_round,
                'indicators': convergence_indicators
            })
        
        # Print impact summary for recent rounds
        if len(self.attack_impact_log['attack_effectiveness_history']) > 1:
            self._print_attack_impact_summary()
    
    def _calculate_convergence_indicators(self):
        """Calculate model convergence indicators to assess attack impact."""
        if len(self.attack_impact_log['round_losses']) < 2:
            return None
            
        # Get recent losses
        recent_losses = [entry['loss'] for entry in self.attack_impact_log['round_losses'][-3:]]
        
        # Loss trend (negative = improving, positive = deteriorating)
        if len(recent_losses) >= 2:
            loss_trend = recent_losses[-1] - recent_losses[-2]
        else:
            loss_trend = 0
            
        # Loss variance (higher = less stable)
        loss_variance = 0
        if len(recent_losses) > 1:
            mean_loss = sum(recent_losses) / len(recent_losses)
            loss_variance = sum((loss - mean_loss) ** 2 for loss in recent_losses) / len(recent_losses)
        
        # Gradient norm trend
        grad_norm_trend = 0
        if len(self.attack_impact_log['gradient_norms']) >= 2:
            recent_norms = [entry['grad_norm'] for entry in self.attack_impact_log['gradient_norms'][-2:]]
            grad_norm_trend = recent_norms[-1] - recent_norms[-2]
        
        return {
            'loss_trend': loss_trend,
            'loss_variance': loss_variance,
            'gradient_norm_trend': grad_norm_trend,
            'convergence_disruption_score': abs(loss_trend) + loss_variance
        }
    
    def _print_attack_impact_summary(self):
        """Print a summary of attack impact over recent rounds."""
        if len(self.attack_impact_log['attack_effectiveness_history']) < 2:
            return
            
        print(f"\n  Multi-Round Attack Impact Summary:")
        
        # Show trend in attack effectiveness
        recent_scores = [entry['metrics']['attack_score'] for entry in self.attack_impact_log['attack_effectiveness_history'][-3:]]
        if len(recent_scores) >= 2:
            score_trend = recent_scores[-1] - recent_scores[-2]
            trend_arrow = "↑" if score_trend > 0.01 else "↓" if score_trend < -0.01 else "→"
            print(f"  - Attack Effectiveness Trend: {trend_arrow} ({score_trend:+.3f})")
        
        # Show impact on convergence if we have loss data
        if self.attack_impact_log['model_convergence_indicators']:
            latest_indicators = self.attack_impact_log['model_convergence_indicators'][-1]['indicators']
            disruption_score = latest_indicators['convergence_disruption_score']
            
            if disruption_score > 0.1:
                disruption_level = "HIGH"
            elif disruption_score > 0.05:
                disruption_level = "MEDIUM" 
            else:
                disruption_level = "LOW"
                
            print(f"  - Convergence Disruption: {disruption_level} (score: {disruption_score:.4f})")
            print(f"  - Loss Trend: {latest_indicators['loss_trend']:+.6f}")
    
    def _save_transformation_log(self, attack_name, original_stats, poisoned_stats, effectiveness_metrics=None):
        """Save detailed transformation log to file."""
        try:
            import json
            import os
            
            round_num = getattr(self, '_current_round', 0)
            log_dir = os.path.join(self.save_results, 'attack_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"{self.id}_{attack_name.lower()}_round_{round_num}.json")
            
            # Find first layer for detailed value comparison
            first_layer = None
            value_changes = {}
            if 'raw_values' in original_stats and 'raw_values' in poisoned_stats:
                first_layer_candidates = [name for name in original_stats['raw_values'].keys()]
                if first_layer_candidates:
                    first_layer = first_layer_candidates[0]
                    orig_vals = original_stats['raw_values'][first_layer]
                    pois_vals = poisoned_stats['raw_values'][first_layer]
                    
                    # Calculate opposite values and analyze inversion quality
                    opposite_vals = [-val for val in orig_vals[:10]]
                    inversion_scores = []
                    change_types = []
                    
                    for i in range(min(10, len(orig_vals))):
                        orig_val = orig_vals[i]
                        pois_val = pois_vals[i]
                        opposite_val = -orig_val
                        
                        # Calculate how close poisoned value is to perfect opposite
                        if abs(orig_val) > 1e-8:
                            inversion_score = abs(pois_val - opposite_val) / abs(orig_val)
                        else:
                            inversion_score = float('inf') if abs(pois_val) > 1e-8 else 0.0
                        
                        inversion_scores.append(inversion_score)
                        
                        # Determine change type
                        if orig_val * pois_val < 0:
                            if inversion_score < 0.1:
                                change_types.append("INV")  # Close to perfect inverse
                            else:
                                change_types.append("FLIP") # Sign flipped but not perfect inverse
                        elif abs(orig_val - pois_val) > 1e-8:
                            change_types.append("MOD")   # Modified but same sign
                        else:
                            change_types.append("NONE")  # No change
                    
                    value_changes[first_layer] = {
                        'original_values': orig_vals[:10],
                        'poisoned_values': pois_vals[:10],
                        'perfect_opposite_values': opposite_vals,
                        'inversion_scores': inversion_scores,  # Lower = closer to perfect opposite
                        'change_types': change_types,
                        'flipped_indices': [i for i in range(min(10, len(orig_vals))) if orig_vals[i] * pois_vals[i] < 0],
                        'inverted_indices': [i for i, score in enumerate(inversion_scores) if score < 0.1 and orig_vals[i] * pois_vals[i] < 0],
                        'modified_indices': [i for i in range(min(10, len(orig_vals))) if abs(orig_vals[i] - pois_vals[i]) > 1e-8],
                        'inversion_quality': {
                            'average_inversion_score': sum(inversion_scores) / len(inversion_scores) if inversion_scores else 0,
                            'perfect_inversions': sum(1 for score in inversion_scores if score < 0.1),
                            'sign_flips': sum(1 for i in range(len(orig_vals[:10])) if orig_vals[i] * pois_vals[i] < 0),
                            'total_values': len(orig_vals[:10])
                        }
                    }

            log_data = {
                'client_id': self.id,
                'attack_type': attack_name,
                'round': round_num,
                'attack_config': self.attack_strategy.get_attack_info() if hasattr(self.attack_strategy, 'get_attack_info') else {},
                'effectiveness_metrics': effectiveness_metrics or {},
                'attack_impact_summary': self.attack_impact_log,
                'transformation': {
                    'original_stats': {k: v for k, v in original_stats.items() if k != 'raw_values'},  # Exclude raw values to keep size manageable
                    'poisoned_stats': {k: v for k, v in poisoned_stats.items() if k != 'raw_values'},
                    'global_norm_change': {
                        'before': original_stats['global']['total_norm'],
                        'after': poisoned_stats['global']['total_norm'],
                        'change_percent': ((poisoned_stats['global']['total_norm'] - original_stats['global']['total_norm']) / original_stats['global']['total_norm'] * 100) if original_stats['global']['total_norm'] > 0 else 0
                    },
                    'detailed_value_changes': value_changes
                }
            }
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
            print(f"[Client {self.id}] 📝 Attack log saved to {log_file}")
            
        except Exception as e:
            print(f"[Client {self.id}] Warning: Could not save attack log: {e}")
            
    def get_attack_info(self):
        """Get information about the current attack configuration."""
        if not self.is_malicious or not self.attack_strategy:
            return None
            
        return self.attack_strategy.get_attack_info()