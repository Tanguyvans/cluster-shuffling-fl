"""
Data loading utilities for attacks
"""

import os
import torch


def find_latest_experiment(results_dir="results"):
    """Find the latest experiment directory"""
    if not os.path.exists(results_dir):
        raise FileNotFoundError("No results directory found. Run main.py first!")
    
    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))]
    
    if not exp_dirs:
        raise FileNotFoundError("No experiment directories found in results/")
    
    # Sort by modification time, get most recent
    exp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_exp = exp_dirs[0]
    
    print(f"Using latest experiment: {latest_exp}")
    return os.path.join(results_dir, latest_exp)


def load_gradient_data(exp_dir, round_num, client_id):
    """Load gradient data from ModelManager format"""
    # ModelManager saves gradients as: models/clients/round_XXX/client_id_gradients.pt
    gradient_path = os.path.join(exp_dir, "models", "clients", f"round_{round_num:03d}", f"{client_id}_gradients.pt")
    
    if not os.path.exists(gradient_path):
        raise FileNotFoundError(f"Gradient file not found: {gradient_path}")
    
    print(f"Loading gradients from: {gradient_path}")
    data = torch.load(gradient_path, map_location='cpu')
    return data


def load_model_data(exp_dir, round_num, client_id):
    """Load model data from ModelManager format"""
    model_path = os.path.join(exp_dir, "models", "clients", f"round_{round_num:03d}", f"{client_id}_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data = torch.load(model_path, map_location='cpu')
    return data


def find_available_clients(exp_dir, attack_rounds):
    """Find all clients with saved gradients for the specified rounds"""
    clients_dir = os.path.join(exp_dir, "models", "clients")
    available_clients = set()
    
    # Scan for available gradient files
    for round_num in attack_rounds:
        round_dir = os.path.join(clients_dir, f"round_{round_num:03d}")
        if os.path.exists(round_dir):
            for file in os.listdir(round_dir):
                if file.endswith('_gradients.pt'):
                    client_id = file.replace('_gradients.pt', '')
                    available_clients.add(client_id)
    
    return sorted(list(available_clients))