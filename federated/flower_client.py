from collections import OrderedDict
import torch
import flwr as fl
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.data_loader import DPDataLoader

from models import Net

from data.loaders import TensorDataset, DataLoader
from utils.device import choice_device
from utils.optimization import fct_loss, choice_optimizer_fct, choice_scheduler_fct
from utils.visualization import save_graphs, save_matrix, save_roc
from utils.model_utils import get_parameters, set_parameters
from core.training import train, test
from security.secret_sharing import apply_smpc, sum_shares

import os


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, epochs=1, model_choice="simpleNet", dp=False, delta=1e-5, epsilon=0.5,
                 max_grad_norm=1.2, device="gpu", classes=(*range(10),),
                 learning_rate=0.001, choice_loss="cross_entropy", choice_optimizer="Adam", choice_scheduler=None,
                 step_size=5, gamma=0.1,
                 save_figure=None, matrix_path=None, roc_path=None, patience=2, pretrained=True,
                 type_ss="additif", threshold=3, m=3, noise_multiplier=1.0, input_size=(32, 32),
                 single_batch_training=False):

        self.batch_size = batch_size
        self.epochs = epochs
        self.model_choice = model_choice
        self.single_batch_training = single_batch_training
        
        # Force disable DP if explicitly set to False
        self.dp = dp
        if not dp:
            print(f"FlowerClient: Differential Privacy DISABLED for {model_choice}")
        else:
            print(f"FlowerClient: Differential Privacy ENABLED for {model_choice}")
            
        self.delta = delta
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.learning_rate = learning_rate
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.len_train = None
        self.device = choice_device(device)
        self.classes = classes
        self.patience = patience

        # MPC and clustering parameters
        self.type_ss = type_ss
        self.threshold = threshold
        self.m = m
        self.list_shapes = None
        self.frag_weights = []
        self.connections = {}

        # Initialize model with provided input size
        model = Net(num_classes=len(self.classes), arch=self.model_choice, pretrained=pretrained, input_size=input_size)
        
        # Validate model for differential privacy if dp is enabled
        if self.dp:
            try:
                model = ModuleValidator.fix(model)
                print("Model validated for differential privacy")
            except Exception as e:
                print(f"Warning: Model validation failed for differential privacy: {e}")
                self.dp = False
        
        self.model = model.to(self.device)
        self.criterion = fct_loss(choice_loss)
        self.choice_optimizer = choice_optimizer
        self.choice_scheduler = choice_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.privacy_engine = None

        self.save_figure = save_figure
        self.matrix_path = matrix_path
        self.roc_path = roc_path
        # Legacy save_model removed - ModelManager handles all model saving

    @classmethod
    def node(cls, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        obj.test_loader = DataLoader(dataset=test_data, batch_size=kwargs['batch_size'], shuffle=True)
        return obj

    @classmethod
    def client(cls, x_train, y_train, x_val, y_val, x_test, y_test, **kwargs):
        obj = cls(**kwargs)
        # Set data loaders
        train_data = TensorDataset(torch.stack(x_train), torch.tensor(y_train))
        val_data = TensorDataset(torch.stack(x_val), torch.tensor(y_val))
        test_data = TensorDataset(torch.stack(x_test), torch.tensor(y_test))

        obj.len_train = len(y_train)
        obj.train_loader = DataLoader(dataset=train_data, batch_size=kwargs['batch_size'], shuffle=True)
        obj.val_loader = DataLoader(dataset=val_data, batch_size=kwargs['batch_size'], shuffle=True)
        obj.test_loader = DataLoader(dataset=test_data, batch_size=kwargs['batch_size'], shuffle=True)
        return obj

    def get_parameters(self, config):
        return get_parameters(self.model)

    def get_dict_params(self, config):
        return {name: val.cpu().numpy() for name, val in self.model.state_dict().items() if 'bn' not in name}

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, node_id, config):
        self.set_parameters(parameters)
        optimizer = choice_optimizer_fct(self.model, choice_optim=self.choice_optimizer, lr=self.learning_rate,
                                     weight_decay=1e-6)
        
        # Check if gradients should be captured for this training session
        capture_gradients = config.get('capture_gradients', False)
        
        if self.dp:
            try:
                # Create privacy engine
                self.privacy_engine = PrivacyEngine(secure_mode=False)
                
                # Make the model private
                self.model, optimizer, self.train_loader = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )
                print(f"Privacy budget: epsilon={self.epsilon}, delta={self.delta}")
            except Exception as e:
                print(f"Warning: Failed to make model private: {e}")
                self.dp = False
                self.privacy_engine = None

        scheduler = choice_scheduler_fct(optimizer, choice_scheduler=self.choice_scheduler,
                                     step_size=self.step_size, gamma=self.gamma)

        results = train(node_id, self.model, self.train_loader, self.val_loader,
                    self.epochs, self.criterion, optimizer, scheduler, device=self.device,
                    dp=self.dp, delta=self.delta,
                    max_physical_batch_size=int(self.batch_size / 4), privacy_engine=self.privacy_engine,
                    patience=self.patience, save_model=None, single_batch_training=self.single_batch_training,
                    capture_gradients=capture_gradients)

        # Model state is already updated in-place by training, no need to reload from file
        best_parameters = self.get_parameters({})
        self.set_parameters(best_parameters)
        
        # Store captured gradients if available
        if capture_gradients and "gradients" in results:
            self.last_gradients = results["gradients"]
            self.last_batch_images = results["gradient_batch_images"]
            self.last_batch_labels = results["gradient_batch_labels"]
            self.last_loss = results["gradient_loss"]
            self.last_accuracy = results["gradient_accuracy"]
            self.last_grad_norm = results["gradient_norm"]
            print(f"[FlowerClient] Stored gradients for gradient inversion attack evaluation")
        
        # Save results
        if self.save_figure:
            save_graphs(self.save_figure, self.epochs, results)

        # Apply SMPC if we have connections (clustering is active)
        if self.connections:
            encrypted_lists, self.list_shapes = apply_smpc(best_parameters, len(self.connections) + 1, 
                                                         self.type_ss, self.threshold)
            # Keep the last share for this client
            self.frag_weights.append(encrypted_lists.pop())
            return encrypted_lists, {'len_train': self.len_train, **results}
        
        return best_parameters, {'len_train': self.len_train, **results}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy, y_pred, y_true, y_proba = test(self.model, self.test_loader, self.criterion, device=self.device)
        if self.save_figure:
            os.makedirs(self.save_figure, exist_ok=True)
            if self.matrix_path:
                save_matrix(y_true, y_pred,
                            self.save_figure + config['name'] + self.matrix_path,
                            self.classes)

            if self.roc_path:
                save_roc(y_true, y_proba,
                         self.save_figure + config['name'] + self.roc_path,  # + f"_client_{self.cid}.png",
                         len(self.classes))

        return {'test_loss': loss, 'test_acc': accuracy}

    def add_connection(self, client_id, address):
        """Add a connection to another client in the same cluster"""
        self.connections[client_id] = {"address": address}

    def reset_connections(self):
        """Reset all client connections (called when clusters change)"""
        self.connections = {}
        self.frag_weights = []

    @property
    def sum_weights(self):
        """Sum the weight shares for MPC"""
        return sum_shares(self.frag_weights, self.type_ss)
