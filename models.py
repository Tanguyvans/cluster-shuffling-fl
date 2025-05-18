import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from typing import Tuple # Added for type hinting

class Net(nn.Module):
    def __init__(self, input_dims: Tuple[int, int, int] = (3, 32, 32), num_classes: int = 10) -> None:
        super(Net, self).__init__()
        # Define the feature extractor part of the network
        # This part's output size will depend on input_dims
        self.features = nn.Sequential(
            nn.Conv2d(input_dims[0], 6, 5), # Use input_dims[0] for input channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate the flattened size of the features output
        # Create a dummy input tensor with the specified dimensions
        # We use torch.no_grad() to ensure this operation doesn't affect gradients
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_dims) # e.g., (1, 3, 32, 32) or (1, 3, 28, 28)
            self._num_flat_features = self.features(dummy_input).view(1, -1).size(1)
            
        # Define the classifier part
        self.fc1 = nn.Linear(self._num_flat_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) # Use num_classes for the output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the tensor dynamically based on its current size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the classifier with a new one for the specified number of classes
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet18(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(MobileNetV2, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # MobileNetV2 uses a 'classifier' attribute which is a Sequential layer
        # The actual Linear layer is the last element in this Sequential layer
        num_ftrs = self.mobilenet_v2.classifier[-1].in_features
        self.mobilenet_v2.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilenet_v2(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(EfficientNetB0, self).__init__()
        self.efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # EfficientNet uses a 'classifier' attribute which is a Sequential layer
        # The actual Linear layer is the last element in this Sequential layer
        num_ftrs = self.efficientnet_b0.classifier[-1].in_features
        self.efficientnet_b0.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet_b0(x)
 
