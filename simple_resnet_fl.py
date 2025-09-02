#!/usr/bin/env python3
"""
Simple federated learning with ResNet18 - save complete models and aggregate properly.
No complex parameter extraction, just save/load complete .pt files and aggregate.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import os

from models.factory import Net
from flwr.server.strategy.aggregate import aggregate

def load_cifar10_data(batch_size=32, num_clients=3):
    """Load CIFAR-10 and split among clients"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='~/data', train=True, download=True, transform=transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=True, transform=transform
    )
    
    # Split training data among clients
    total_size = len(trainset)
    data_per_client = total_size // num_clients
    splits = [data_per_client] * num_clients
    
    # Handle remainder
    remainder = total_size - sum(splits)
    if remainder > 0:
        splits[-1] += remainder  # Give remainder to last client
    
    client_datasets = torch.utils.data.random_split(trainset, splits)
    
    # Create dataloaders
    client_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in client_datasets
    ]
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader

def train_client(model, dataloader, epochs=2, lr=0.01):
    """Train a client model for specified epochs"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    total_loss = 0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            epoch_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_total += target.size(0)
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        epoch_acc = 100. * epoch_correct / epoch_total
        print(f'  Epoch {epoch+1}/{epochs}: Loss={epoch_loss/len(dataloader):.4f}, '
              f'Accuracy={epoch_acc:.2f}%')
    
    return len(dataloader.dataset), epoch_loss / len(dataloader)

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy

def aggregate_models_simple(client_models, dataset_sizes):
    """Simple model aggregation using Flower's aggregate function"""
    print("=== Aggregating Client Models ===")
    
    # Extract parameters from each model
    client_params = []
    for i, model in enumerate(client_models):
        # Get ALL parameters (including BatchNorm) except num_batches_tracked
        params = [param.detach().cpu().numpy() 
                 for name, param in model.state_dict().items() 
                 if 'num_batches_tracked' not in name]
        client_params.append((params, dataset_sizes[i]))
        print(f"Client {i+1}: {len(params)} parameters, {dataset_sizes[i]} samples")
    
    # Use Flower's aggregate function
    aggregated_params = aggregate(client_params)
    print(f"Aggregated: {len(aggregated_params)} parameters")
    
    # Create new global model
    global_model = Net(num_classes=10, arch='resnet18', pretrained=False)
    
    # Set aggregated parameters
    param_names = [name for name in global_model.state_dict().keys() 
                   if 'num_batches_tracked' not in name]
    
    new_state_dict = {}
    for name, param in zip(param_names, aggregated_params):
        new_state_dict[name] = torch.tensor(param)
    
    # Keep num_batches_tracked from original model
    for name in global_model.state_dict().keys():
        if 'num_batches_tracked' in name:
            new_state_dict[name] = global_model.state_dict()[name]
    
    global_model.load_state_dict(new_state_dict, strict=True)
    return global_model

def simple_federated_learning(num_clients=3, num_rounds=3, epochs_per_round=2):
    """Simple federated learning with ResNet18"""
    print("=== Simple Federated Learning with ResNet18 ===")
    
    # Load data
    client_loaders, test_loader = load_cifar10_data(batch_size=32, num_clients=num_clients)
    print(f"Loaded CIFAR-10 data for {num_clients} clients")
    
    # Initialize global model
    global_model = Net(num_classes=10, arch='resnet18', pretrained=False)
    print("Initialized global ResNet18 model")
    
    # Create results directory
    results_dir = Path("simple_fl_results")
    results_dir.mkdir(exist_ok=True)
    
    for round_num in range(num_rounds):
        print(f"\n=== ROUND {round_num + 1} ===")
        
        # Train each client
        client_models = []
        dataset_sizes = []
        
        for client_id in range(num_clients):
            print(f"\nTraining Client {client_id + 1}...")
            
            # Create fresh model copy for this client
            client_model = Net(num_classes=10, arch='resnet18', pretrained=False)
            client_model.load_state_dict(global_model.state_dict())
            
            # Train client model
            dataset_size, final_loss = train_client(
                client_model, client_loaders[client_id], 
                epochs=epochs_per_round
            )
            
            # Save client model
            client_path = results_dir / f"round_{round_num+1}_client_{client_id+1}.pt"
            torch.save({
                'model_state_dict': client_model.state_dict(),
                'dataset_size': dataset_size,
                'final_loss': final_loss,
                'round': round_num + 1,
                'client_id': client_id + 1
            }, client_path)
            
            client_models.append(client_model)
            dataset_sizes.append(dataset_size)
            
            # Evaluate client model
            test_loss, test_acc = evaluate_model(client_model, test_loader)
            print(f"Client {client_id + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Aggregate models
        global_model = aggregate_models_simple(client_models, dataset_sizes)
        
        # Evaluate global model
        global_test_loss, global_test_acc = evaluate_model(global_model, test_loader)
        print(f"\n🌐 Global Model - Test Loss: {global_test_loss:.4f}, Test Accuracy: {global_test_acc:.2f}%")
        
        # Save global model
        global_path = results_dir / f"global_round_{round_num+1}.pt"
        torch.save({
            'model_state_dict': global_model.state_dict(),
            'test_loss': global_test_loss,
            'test_accuracy': global_test_acc,
            'round': round_num + 1
        }, global_path)
        
        print(f"Saved models to {results_dir}/")

if __name__ == "__main__":
    simple_federated_learning(num_clients=3, num_rounds=3, epochs_per_round=2)