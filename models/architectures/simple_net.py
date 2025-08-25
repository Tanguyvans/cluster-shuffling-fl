import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SimpleNet(nn.Module):
    """
    A simple CNN model
    """
    def __init__(self, num_classes=10, input_size=(32, 32)) -> None:
        super(SimpleNet, self).__init__()
        self.input_size = input_size
        # 3 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 6, 5)

        # with Batch Normalization layers for the non-iid data
        # self.bn1 = nn.BatchNorm2d(6)

        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input image channel, 16 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)

        # with Batch Normalization layers
        # self.bn2 = nn.BatchNorm2d(16)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self._get_fc_input_size(), 120)  # 16 * 5 * 5 for an input of 32x32

        # with Batch Normalization layers
        # self.bn3 = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(120, 84)

        # with Batch Normalization layers
        # self.bn4 = nn.BatchNorm1d(84)

        self.fc3 = nn.Linear(84, num_classes)

    def _get_fc_input_size(self):
        """
        Determine the size of the input to the fully connected layers.
        """
        dummy_input = torch.zeros(1, 3, *self.input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        # Calculate the size of the flattened features
        return math.prod(x.size()[1:])  # Exclude the batch dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # output layer

        # Note: the softmax function is not used here because it is included in the loss function
        return x