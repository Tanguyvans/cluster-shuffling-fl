#!/usr/bin/env python3
"""
FFHQ Training Script with ResNet18 for Gradient Inversion Attacks
Trains a ResNet18 classifier on age groups
Prepares training artifacts for gradient inversion attacks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# FFHQ normalization (ImageNet values for ResNet)
FFHQ_MEAN = [0.485, 0.456, 0.406]
FFHQ_STD = [0.229, 0.224, 0.225]


class FFHQAgeDataset(Dataset):
    """
    FFHQ dataset with age-based classification
    Age groups: 0=0-10, 1=10-20, 2=20-30, 3=30-40, 4=40-50, 5=50+
    """
    
    def __init__(self, data_dir, json_dir, transform=None, max_images_per_class=None):
        self.data_dir = Path(data_dir)
        self.json_dir = Path(json_dir)
        self.transform = transform
        
        self.images = []
        self.labels = []
        self.ages = []
        self.image_paths = []
        
        # Age group mapping
        self.age_groups = {
            0: "0-10 years",
            1: "10-20 years", 
            2: "20-30 years",
            3: "30-40 years",
            4: "40-50 years",
            5: "50+ years"
        }
        
        # Load images and create age-based labels
        self._load_dataset(max_images_per_class)
        
        print(f"Loaded {len(self.images)} images")
        self._print_class_distribution()
    
    def _age_to_class(self, age):
        """Convert age to age group class"""
        if age < 10:
            return 0
        elif age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 40:
            return 3
        elif age < 50:
            return 4
        else:
            return 5
    
    def _load_dataset(self, max_images_per_class):
        """Load images and corresponding age labels from JSON files"""
        class_counts = defaultdict(int)
        
        # Get all image files
        image_files = list(self.data_dir.glob("**/*.png")) + list(self.data_dir.glob("**/*.jpg"))
        image_files.sort()
        
        # For single image dataset, just take the first valid image
        if max_images_per_class == 1:
            found_image = False
        
        for img_path in image_files:
            # Extract image ID from filename
            img_id = img_path.stem
            json_path = self.json_dir / f"{img_id}.json"
            
            # For single image mode, don't require JSON - assign default class 0
            if max_images_per_class == 1 and not found_image:
                self.image_paths.append(img_path)
                self.labels.append(0)  # Default to class 0
                self.ages.append(25)   # Default age
                found_image = True
                break
            
            if not json_path.exists():
                continue
            
            # Load age from JSON
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if not data or 'faceAttributes' not in data[0]:
                        continue
                    
                    age = data[0]['faceAttributes']['age']
                    age_class = self._age_to_class(age)
                    
                    # Original logic for multiple images per class
                    if max_images_per_class and class_counts[age_class] >= max_images_per_class:
                        continue
                    
                    self.image_paths.append(img_path)
                    self.labels.append(age_class)
                    self.ages.append(age)
                    class_counts[age_class] += 1
                    
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error loading {json_path}: {e}")
                continue
    
    def _print_class_distribution(self):
        """Print distribution of images across age groups"""
        class_counts = defaultdict(int)
        for label in self.labels:
            class_counts[label] += 1
        
        print("\nAge group distribution:")
        for class_id in sorted(class_counts.keys()):
            print(f"  Class {class_id} ({self.age_groups[class_id]}): {class_counts[class_id]} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_resnet18_model(num_classes=1, weights="DEFAULT", input_size=32):
    """Create ResNet18 model for classification"""
    model = models.resnet18(weights=weights)
    
    # For 32x32 images, we need to modify the first conv layer and remove some pooling
    if input_size == 32:
        # Replace first conv layer (original: 7x7 kernel, stride 2, padding 3)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the first max pooling layer for small images
        model.maxpool = nn.Identity()
    
    # Replace final layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_dataloaders(data_dir, json_dir, batch_size=1, input_size=32, 
                      train_ratio=0.8, max_images_per_class=None):
    """Create train and validation dataloaders"""
    
    # Data transforms - Modified for 32x32 images to match GIAS core
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=FFHQ_MEAN, std=FFHQ_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=FFHQ_MEAN, std=FFHQ_STD)
    ])
    
    # Load full dataset
    full_dataset = FFHQAgeDataset(data_dir, json_dir, transform=None, 
                                 max_images_per_class=max_images_per_class)
    
    if len(full_dataset) == 0:
        raise ValueError("No valid images found in dataset!")
    
    # For single image, use the same image for both train and validation
    total_size = len(full_dataset)
    if total_size == 1:
        train_indices = [0]
        val_indices = [0]
    else:
        train_size = max(1, int(train_ratio * total_size))
        # Create indices for splitting
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:] if train_size < total_size else [0]
    
    # Create separate datasets with transforms
    train_dataset = FFHQAgeDataset(data_dir, json_dir, transform=train_transform,
                                  max_images_per_class=max_images_per_class)
    val_dataset = FFHQAgeDataset(data_dir, json_dir, transform=val_transform,
                                max_images_per_class=max_images_per_class)
    
    # Filter datasets by indices
    train_dataset.image_paths = [train_dataset.image_paths[i] for i in train_indices]
    train_dataset.labels = [train_dataset.labels[i] for i in train_indices]
    train_dataset.ages = [train_dataset.ages[i] for i in train_indices]
    
    val_dataset.image_paths = [val_dataset.image_paths[i] for i in val_indices]
    val_dataset.labels = [val_dataset.labels[i] for i in val_indices]
    val_dataset.ages = [val_dataset.ages[i] for i in val_indices]
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, train_dataset.age_groups


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def save_training_artifacts(model, train_loader, age_groups, device, output_dir="./results/ffhq_resnet_training"):
    """Save training artifacts for gradient inversion attacks"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving training artifacts for gradient inversion...")
    
    # Get a small batch for gradient extraction
    model.eval()
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    # Use only 1 image as requested
    batch_size = min(1, images.size(0))
    images = images[:batch_size].to(device)
    labels = labels[:batch_size].to(device)
    
    print(f"Using batch of {batch_size} image for attack preparation")
    print(f"Labels: {labels.tolist()}")
    print(f"Age groups: {[age_groups[label.item()] for label in labels]}")
    
    # Forward pass to compute loss
    outputs = model(images)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(outputs, labels)
    
    # Compute gradients
    gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
    gradients = [g.detach().cpu() for g in gradients]
    
    # Denormalize images for saving
    mean = torch.tensor(FFHQ_MEAN).view(3, 1, 1)
    std = torch.tensor(FFHQ_STD).view(3, 1, 1)
    images_denorm = images.cpu() * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Save training configuration
    config = {
        'model_name': 'ResNet18',
        'num_classes': len(age_groups),
        'input_size': 32,  # Changed to 32x32 to match GIAS core
        'age_groups': age_groups,
        'normalization': {
            'mean': FFHQ_MEAN,
            'std': FFHQ_STD
        },
        'batch_size': batch_size,
        'dataset': 'FFHQ'
    }
    
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save model
    model_data = {
        'model_state_dict': model.state_dict(),
        'num_classes': len(age_groups),
        'input_size': 32,
        'architecture': 'ResNet18'
    }
    torch.save(model_data, output_dir / 'trained_model.pth')
    
    # Save gradients
    gradient_data = {
        'gradients': gradients,
        'loss': loss.item()
    }
    torch.save(gradient_data, output_dir / 'gradients.pth')
    
    # Save training data (ground truth)
    training_data = {
        'images_raw': images_denorm,
        'labels': labels.cpu(),
        'age_groups': [age_groups[label.item()] for label in labels]
    }
    torch.save(training_data, output_dir / 'training_data.pth')
    
    print(f"✓ Training artifacts saved to {output_dir}")
    return output_dir


def train_ffhq_resnet():
    """Main training function with ResNet18"""
    print("=" * 80)
    print("   FFHQ RESNET18 TRAINING")
    print("=" * 80)
    
    # Configuration
    config = {
        'data_dir': './data/ffhq_dataset',
        'json_dir': './data/json',
        'batch_size': 1,  # Batch size of 1 for single image
        'input_size': 32,  # Changed to 32x32 to match GIAS core FFHQ dataset
        'num_epochs': 1,  # Single epoch for single image
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'max_images_per_class': 1,  # Exactly 1 total image
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"Device: {config['device']}")
    print(f"Max images per class: {config['max_images_per_class']}")
    print(f"Input size: {config['input_size']}x{config['input_size']}")
    
    # Create dataloaders
    print("\nLoading FFHQ dataset...")
    train_loader, val_loader, age_groups = create_dataloaders(
        config['data_dir'], config['json_dir'], 
        batch_size=config['batch_size'],
        input_size=config['input_size'],
        max_images_per_class=config['max_images_per_class']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create ResNet18 model
    model = create_resnet18_model(num_classes=len(age_groups), weights="DEFAULT", input_size=config['input_size'])
    model = model.to(config['device'])
    
    print(f"\nModel: ResNet18")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config['device'])
        
        # Update scheduler
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training artifacts for gradient inversion
    artifacts_dir = save_training_artifacts(model, train_loader, age_groups, config['device'])
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Acc')
    plt.plot(val_accs, 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(artifacts_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 Ready for gradient inversion attacks!")
    print(f"   Run: python examples/attacks/gias/gias_ffhq_attack.py")
    
    return model, age_groups


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        model, age_groups = train_ffhq_resnet()
        print("\n✓ ResNet18 training completed successfully!")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()