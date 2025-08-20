"""
Federated Learning with Single Batch Training and Gradient Saving for Inversion Attacks
Integrates with the existing cluster-shuffling-fl framework
"""

import torch
import torch.nn as nn
import numpy as np
import os
import copy
import sys
from typing import Dict, List, Tuple, Optional
import inversefed

# Import from existing framework
from models.factory import Net
from data.loaders import load_data_from_path
import config as cfg

class SingleBatchClient:
    """Client that trains on a single batch (one image per class) for gradient inversion testing"""
    
    def __init__(self, client_id: int, global_model: nn.Module, dataset, device: torch.device, 
                 num_classes: int = 10, seed_offset: int = 0):
        self.client_id = client_id
        self.model = copy.deepcopy(global_model)
        self.device = device
        self.num_classes = num_classes
        
        # Loss function - using CrossEntropyLoss like the main framework
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer - matching main framework settings
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=cfg.settings['lr']
        )
        
        # Select one image per class
        self.batch_indices, self.batch_labels_list = self._select_one_per_class(dataset, seed_offset)
        
        # Extract and prepare the batch
        self._prepare_batch(dataset)
        
        print(f"Client {client_id}: Selected indices {self.batch_indices[:5]}..., classes {sorted(set(self.batch_labels_list))}")
    
    def _select_one_per_class(self, dataset, seed_offset: int) -> Tuple[List[int], List[int]]:
        """Select one image per class for unique labels"""
        np.random.seed(42 + seed_offset)
        selected_indices = []
        selected_labels = []
        class_counts = {i: 0 for i in range(self.num_classes)}
        
        # Shuffle indices
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)
        
        for idx in all_indices:
            _, label = dataset[idx]
            if class_counts[label] == 0:
                selected_indices.append(idx)
                selected_labels.append(label)
                class_counts[label] = 1
                if len(selected_indices) == self.num_classes:
                    break
        
        return selected_indices, selected_labels
    
    def _prepare_batch(self, dataset):
        """Prepare the batch tensors"""
        images = []
        labels = []
        
        for idx in self.batch_indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)
        
        self.batch_images = torch.stack(images).to(self.device)
        self.batch_labels = torch.tensor(labels).to(self.device)
    
    def update_model(self, global_state_dict: Dict):
        """Update client model with global model state"""
        self.model.load_state_dict(global_state_dict)
    
    def train_local(self, epochs: int, return_gradients: bool = False) -> Dict:
        """Train locally on the single batch"""
        self.model.train()
        
        training_losses = []
        training_accuracies = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.batch_images)
            loss = self.loss_fn(outputs, self.batch_labels)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(self.batch_labels).sum().item()
            accuracy = 100. * correct / len(self.batch_labels)
            
            training_losses.append(loss.item())
            training_accuracies.append(accuracy)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.batch_images)
            final_loss = self.loss_fn(outputs, self.batch_labels)
            _, predicted = outputs.max(1)
            correct = predicted.eq(self.batch_labels).sum().item()
            final_accuracy = 100. * correct / len(self.batch_labels)
        
        # Get gradients if requested
        gradients = None
        if return_gradients:
            self.model.zero_grad()
            outputs = self.model(self.batch_images)
            loss = self.loss_fn(outputs, self.batch_labels)
            
            # Get parameter gradients
            gradients = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            gradients = [grad.detach().clone() for grad in gradients]
        
        return {
            'client_id': self.client_id,
            'final_loss': final_loss.item(),
            'final_accuracy': final_accuracy,
            'avg_loss': np.mean(training_losses),
            'avg_accuracy': np.mean(training_accuracies),
            'gradients': gradients,
            'model_state': copy.deepcopy(self.model.state_dict())
        }


def federated_averaging(client_updates: List[Dict], global_model: nn.Module) -> nn.Module:
    """Simple federated averaging"""
    global_dict = global_model.state_dict()
    
    # Average all client model parameters
    for key in global_dict.keys():
        stacked = torch.stack([
            client_updates[i]['model_state'][key].float() 
            for i in range(len(client_updates))
        ])
        global_dict[key] = stacked.mean(dim=0).to(global_dict[key].dtype)
    
    global_model.load_state_dict(global_dict)
    return global_model


def evaluate_global_model(global_model: nn.Module, clients: List[SingleBatchClient]) -> Tuple[float, float]:
    """Evaluate global model on all client batches"""
    global_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for client in clients:
            outputs = global_model(client.batch_images)
            loss = client.loss_fn(outputs, client.batch_labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(client.batch_labels).sum().item()
            total += client.batch_labels.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(clients)
    return avg_loss, accuracy


def main():
    """Federated learning with single batch training for gradient inversion attacks"""
    
    # Configuration - use existing config with some overrides for single batch testing
    NUM_CLIENTS = min(cfg.settings['number_of_clients_per_node'], 2)  # Limit for single batch testing
    FEDERATED_ROUNDS = cfg.settings['n_rounds']
    EPOCHS_PER_CLIENT = cfg.settings['n_epochs']
    SAVE_GRADIENTS_ROUNDS = [1, 5, 10] if FEDERATED_ROUNDS >= 10 else [1, FEDERATED_ROUNDS]
    
    print("=== Federated Learning with Single Batch Training (Gradient Inversion Ready) ===")
    print(f"Device: cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {cfg.settings['arch']}")
    print(f"Dataset: {cfg.settings['name_dataset']}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {FEDERATED_ROUNDS}")
    print(f"Epochs per client: {EPOCHS_PER_CLIENT}")
    print(f"Save gradients at rounds: {SAVE_GRADIENTS_ROUNDS}")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset using existing framework
    train_dataset, test_dataset = load_data_from_path(
        name_dataset=cfg.settings['name_dataset'],
        data_root=cfg.settings['data_root']
    )
    
    # Determine number of classes
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    elif cfg.settings['name_dataset'] == 'cifar10':
        num_classes = 10
    elif cfg.settings['name_dataset'] == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10  # default
    
    # Use test dataset for controlled single-batch training
    dataset = test_dataset
    
    # Create global model using existing framework
    global_model = Net(
        num_classes=num_classes,
        arch=cfg.settings['arch'],
        pretrained=False
    ).to(device)
    
    print(f"Model: {cfg.settings['arch']}")
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create clients
    clients = []
    for client_id in range(NUM_CLIENTS):
        client = SingleBatchClient(
            client_id=client_id,
            global_model=global_model,
            dataset=dataset,
            device=device,
            num_classes=num_classes,
            seed_offset=client_id
        )
        clients.append(client)
    
    # Create directories for saving gradients and models
    gradient_dir = f'results/gradient_inversion/{cfg.settings["arch"]}_{cfg.settings["name_dataset"]}'
    model_dir = f'results/gradient_inversion/models/{cfg.settings["arch"]}_{cfg.settings["name_dataset"]}'
    os.makedirs(gradient_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Starting Federated Learning with Gradient Saving")
    print(f"{'='*70}")
    
    # Training history
    history = {
        'rounds': [],
        'global_loss': [],
        'global_accuracy': [],
        'client_losses': [],
        'client_accuracies': []
    }
    
    # Federated training loop
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{FEDERATED_ROUNDS} ---")
        
        client_updates = []
        save_gradients = round_num in SAVE_GRADIENTS_ROUNDS
        
        # Each client trains locally
        for client in clients:
            # Update with latest global model
            client.update_model(global_model.state_dict())
            
            # Train locally
            update = client.train_local(EPOCHS_PER_CLIENT, return_gradients=save_gradients)
            client_updates.append(update)
            
            print(f"  Client {client.client_id}: Loss={update['final_loss']:.4f}, "
                  f"Acc={update['final_accuracy']:.2f}%")
            
            # Save gradients and training data if needed
            if save_gradients and update['gradients'] is not None:
                # Calculate gradient norm for vulnerability analysis
                grad_norm = torch.stack([g.norm() for g in update['gradients']]).mean().item()
                
                save_data = {
                    'round': round_num,
                    'client_id': client.client_id,
                    'gradients': [g.cpu() for g in update['gradients']],
                    'loss': update['final_loss'],
                    'accuracy': update['final_accuracy'],
                    'grad_norm': grad_norm,
                    'model_state': update['model_state'],
                    'batch_indices': client.batch_indices,
                    'batch_labels': client.batch_labels.cpu(),
                    'batch_images': client.batch_images.cpu(),
                    'model_architecture': cfg.settings['arch'],
                    'num_classes': num_classes,
                    'dataset': cfg.settings['name_dataset']
                }
                
                filename = f'{gradient_dir}/round_{round_num}_client_{client.client_id}.pt'
                torch.save(save_data, filename)
        
        # Federated averaging
        global_model = federated_averaging(client_updates, global_model)
        
        # Evaluate global model
        global_loss, global_accuracy = evaluate_global_model(global_model, clients)
        
        # Calculate averages
        avg_client_loss = np.mean([u['final_loss'] for u in client_updates])
        avg_client_accuracy = np.mean([u['final_accuracy'] for u in client_updates])
        
        print(f"  Global Model: Loss={global_loss:.4f}, Acc={global_accuracy:.2f}%")
        print(f"  Avg Client: Loss={avg_client_loss:.4f}, Acc={avg_client_accuracy:.2f}%")
        
        # Update history
        history['rounds'].append(round_num)
        history['global_loss'].append(global_loss)
        history['global_accuracy'].append(global_accuracy)
        history['client_losses'].append(avg_client_loss)
        history['client_accuracies'].append(avg_client_accuracy)
        
        # Save global model for key rounds
        if save_gradients or round_num == FEDERATED_ROUNDS:
            torch.save({
                'round': round_num,
                'model_state': global_model.state_dict(),
                'global_loss': global_loss,
                'global_accuracy': global_accuracy,
                'arch': cfg.settings['arch'],
                'num_classes': num_classes,
                'dataset': cfg.settings['name_dataset']
            }, f'{model_dir}/global_model_round_{round_num}.pt')
        
        if save_gradients:
            print(f"  ✅ Saved gradients and models for round {round_num}")
        
        sys.stdout.flush()
    
    # Save training history
    torch.save(history, f'{gradient_dir}/training_history.pt')
    
    print(f"\n{'='*70}")
    print("Federated Learning Complete!")
    print(f"{'='*70}")
    print(f"Final global accuracy: {global_accuracy:.2f}%")
    print(f"Gradients saved for rounds: {SAVE_GRADIENTS_ROUNDS}")
    print(f"Results saved in: {gradient_dir}")
    
    # Print summary for gradient inversion attack
    print(f"\n{'='*50}")
    print("Ready for Gradient Inversion Attack")
    print(f"{'='*50}")
    print(f"Gradient files saved: {len(SAVE_GRADIENTS_ROUNDS) * NUM_CLIENTS}")
    print(f"Run 'python gradient_inversion_attack.py' to start the attack")
    print(f"Files are compatible with your existing attack script")
    
    return gradient_dir, model_dir


if __name__ == "__main__":
    # Set deterministic behavior for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    gradient_dir, model_dir = main()