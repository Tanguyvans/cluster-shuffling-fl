import torch.nn as nn
import torch
from .architectures.simple_net import SimpleNet
from .architectures.efficient_net import EfficientNet
from .architectures.mobilenet import MobileNet
from .architectures.squeeze_net import SqueezeNet
from .architectures.resnet import ResNet
from .architectures.shuffle_net import ShuffleNet

class Net(nn.Module):
    """
    This is a generic class to choose the architecture of the model.
    param num_classes: the number of classes
    param arch: the architecture of the model (str)
    param input_size: the input image size (height, width)

    return: the model
    """
    def __init__(self, num_classes=10, arch="simpleNet", pretrained=True, input_size=(32, 32)) -> None:
        super(Net, self).__init__()
        print("Number of classes : ", num_classes, " and the architecture is : ", arch)
        
        if "simplenet" in arch.lower():
            self.model = SimpleNet(num_classes=num_classes, input_size=input_size)
        elif "efficientnet" in arch.lower():
            self.model = EfficientNet(num_classes=num_classes, arch=arch, pretrained=pretrained)
        elif "mobilenet" in arch.lower():
            self.model = MobileNet(num_classes=num_classes, pretrained=pretrained)
        elif "squeezenet" in arch.lower():
            self.model = SqueezeNet(num_classes=num_classes, pretrained=pretrained)
        elif "resnet" in arch.lower():
            self.model = ResNet(num_classes=num_classes, arch=arch, pretrained=pretrained)
        elif "shufflenet" in arch.lower():
            size = '1.0x'
            if '_' in arch:
                size = arch.split('_')[1]
            self.model = ShuffleNet(num_classes=num_classes, size=size, pretrained=pretrained)
        else:
            raise NotImplementedError("The architecture is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        return self.model(x)