"""
ConvNet model replicated from inversefed for gradient inversion attacks.
Original implementation from inversefed.nn.models.py
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class ConvNet(torch.nn.Module):
    """ConvNetBN - A convolutional network with batch normalization designed for gradient inversion attacks."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """
        Initialize ConvNet.
        
        Args:
            width: Base width multiplier for channels (default: 32, inversefed uses 64)
            num_classes: Number of output classes
            num_channels: Number of input channels (3 for RGB images)
        """
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('conv8', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn8', torch.nn.BatchNorm2d(4 * width)),
            ('relu8', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten()),
            ('linear', torch.nn.Linear(36 * width, num_classes))
        ]))

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


def convnet(num_classes=10, num_channels=3, width=64, **kwargs):
    """
    Create ConvNet model with standard inversefed parameters.
    
    Args:
        num_classes: Number of output classes
        num_channels: Number of input channels  
        width: Base width multiplier (64 matches inversefed default)
        **kwargs: Additional arguments (ignored)
        
    Returns:
        ConvNet model instance
    """
    return ConvNet(width=width, num_classes=num_classes, num_channels=num_channels)