"""
Simple Federated Learning with 2 clients
Each client trains on a single batch of 10 images (1 per class) like train.py
"""

import torch
import numpy as np
import os
import copy
import inversefed
from models import Net  # Import your custom models

# Configuration
MODEL_NAME = 'ConvNet'
DATASET = 'CIFAR10'
NUM_CLIENTS = 2
FEDERATED_ROUNDS = 10
EPOCHS_PER_CLIENT = 3
BATCH_SIZE = 10  # One image per class
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_GRADIENTS_ROUNDS = [1, 5, 10]

def select_one_per_class(dataset, num_classes=10, seed_offset=0):
    """Select one image per class for unique labels"""
    np.random.seed(42 + seed_offset)  # Different seed for each client
    selected_indices = []
    selected_labels = set()
    
    # Shuffle indices to get different images for each client
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    
    for idx in all_indices:
        _, label = dataset[idx]
        if label not in selected_labels:
            selected_indices.append(idx)
            selected_labels.add(label)
            if len(selected_indices) == num_classes:
                break
    
    return selected_indices, sorted(selected_labels)

class FederatedClient:
    def __init__(self, client_id, global_model, dataset, setup, loss_fn):
        self.client_id = client_id
        self.model = copy.deepcopy(global_model)
        self.setup = setup
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        # Select unique batch for this client
        self.batch_indices, self.labels = select_one_per_class(dataset, seed_offset=client_id)
        
        # Extract the batch
        images = []
        labels = []
        for idx in self.batch_indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)
        
        # Convert to tensors and move to device
        self.batch_images = torch.stack(images).to(**setup)
        self.batch_labels = torch.tensor(labels).to(device=setup['device'], dtype=torch.long)
        
        print(f"Client {client_id}: indices {self.batch_indices[:5]}..., classes {self.labels}")
    
    def update_model(self, global_model):
        """Update client model with global model state"""
        self.model.load_state_dict(global_model.state_dict())
    
    def train_local(self, epochs, return_gradients=False, use_gradient_aggregation=False):
        """Train locally on the single batch for specified epochs"""
        self.model.train()
        
        training_losses = []
        training_accuracies = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.batch_images)
            loss, _, _ = self.loss_fn(outputs, self.batch_labels)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(self.batch_labels).sum().item()
            accuracy = 100. * correct / BATCH_SIZE
            
            training_losses.append(loss.item())
            training_accuracies.append(accuracy)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.batch_images)
            final_loss, _, _ = self.loss_fn(outputs, self.batch_labels)
            _, predicted = outputs.max(1)
            correct = predicted.eq(self.batch_labels).sum().item()
            final_accuracy = 100. * correct / BATCH_SIZE
        
        # Always compute gradients for gradient-based aggregation
        gradients = None
        self.model.zero_grad()
        outputs = self.model(self.batch_images)
        loss, _, _ = self.loss_fn(outputs, self.batch_labels)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        gradients = [grad.detach().clone() for grad in gradients]
        
        # For gradient-based aggregation, return gradients instead of model weights
        if use_gradient_aggregation:
            return {
                'client_id': self.client_id,
                'final_loss': final_loss.item(),
                'final_accuracy': final_accuracy,
                'avg_loss': np.mean(training_losses),
                'avg_accuracy': np.mean(training_accuracies),
                'gradients': gradients,
                'model_state': copy.deepcopy(self.model.state_dict()),
                'aggregation_data': gradients  # Use gradients for aggregation
            }
        else:
            return {
                'client_id': self.client_id,
                'final_loss': final_loss.item(),
                'final_accuracy': final_accuracy,
                'avg_loss': np.mean(training_losses),
                'avg_accuracy': np.mean(training_accuracies),
                'gradients': gradients if return_gradients else None,
                'model_state': copy.deepcopy(self.model.state_dict()),
                'aggregation_data': copy.deepcopy(self.model.state_dict())  # Use model weights for aggregation
            }

def federated_averaging(client_updates, global_model):
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

def evaluate_global_model(global_model, clients, loss_fn):
    """Evaluate global model on all client batches"""
    global_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for client in clients:
            outputs = global_model(client.batch_images)
            loss, _, _ = loss_fn(outputs, client.batch_labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(client.batch_labels).sum().item()
            total += client.batch_labels.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(clients)
    return avg_loss, accuracy

def main():
    """Simple federated learning with 2 clients"""
    print("=== Simple Federated Learning (2 Clients, Single Batch) ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {FEDERATED_ROUNDS}")
    print(f"Epochs per client: {EPOCHS_PER_CLIENT}")
    print()
    
    # Setup
    inversefed.utils.set_deterministic()
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load data
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        DATASET, defs, data_path='./datasets/cifar10'
    )
    
    # Create global model using your custom Net class
    global_model = Net(num_classes=10, arch='convnet', pretrained=False, input_size=(32, 32))
    global_model.to(**setup)
    
    print(f"Model: {MODEL_NAME}")
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create clients
    clients = []
    for client_id in range(NUM_CLIENTS):
        client = FederatedClient(
            client_id=client_id,
            global_model=global_model,
            dataset=validloader.dataset,
            setup=setup,
            loss_fn=loss_fn
        )
        clients.append(client)
    
    # Create directories
    os.makedirs('federated_simple_gradients', exist_ok=True)
    os.makedirs('federated_simple_models', exist_ok=True)
    
    print(f"\n{'='*50}")
    print("Starting Federated Learning")
    print(f"{'='*50}")
    
    # Federated training loop
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{FEDERATED_ROUNDS} ---")
        
        client_updates = []
        save_gradients = round_num in SAVE_GRADIENTS_ROUNDS
        
        # Each client trains locally
        for client in clients:
            # Update with latest global model
            client.update_model(global_model)
            
            # Train locally
            update = client.train_local(EPOCHS_PER_CLIENT, return_gradients=save_gradients)
            client_updates.append(update)
            
            print(f"  Client {client.client_id}: Loss={update['final_loss']:.4f}, Acc={update['final_accuracy']:.2f}%")
            
            # Save gradients if needed
            if save_gradients and update['gradients'] is not None:
                torch.save({
                    'round': round_num,
                    'client_id': client.client_id,
                    'gradients': [g.cpu() for g in update['gradients']],
                    'loss': update['final_loss'],
                    'accuracy': update['final_accuracy'],
                    'model_state': update['model_state'],
                    'batch_indices': client.batch_indices,
                    'batch_labels': client.batch_labels.cpu()
                }, f'federated_simple_gradients/round_{round_num}_client_{client.client_id}.pt')
        
        # Federated averaging
        global_model = federated_averaging(client_updates, global_model)
        
        # Evaluate global model
        global_loss, global_accuracy = evaluate_global_model(global_model, clients, loss_fn)
        
        # Calculate averages
        avg_client_loss = np.mean([u['final_loss'] for u in client_updates])
        avg_client_accuracy = np.mean([u['final_accuracy'] for u in client_updates])
        
        print(f"  Global Model: Loss={global_loss:.4f}, Acc={global_accuracy:.2f}%")
        print(f"  Avg Client: Loss={avg_client_loss:.4f}, Acc={avg_client_accuracy:.2f}%")
        
        # Save global model
        torch.save({
            'round': round_num,
            'model_state': global_model.state_dict(),
            'global_loss': global_loss,
            'global_accuracy': global_accuracy
        }, f'federated_simple_models/global_model_round_{round_num}.pt')
        
        if save_gradients:
            print(f"  ✅ Saved gradients for round {round_num}")
    
    print(f"\n{'='*50}")
    print("Federated Learning Complete!")
    print(f"{'='*50}")
    print(f"Final global accuracy: {global_accuracy:.2f}%")
    print(f"Gradients saved for rounds: {SAVE_GRADIENTS_ROUNDS}")

if __name__ == "__main__":
    main()