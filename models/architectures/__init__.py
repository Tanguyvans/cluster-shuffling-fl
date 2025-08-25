from .simple_net import SimpleNet
from .efficient_net import EfficientNet
from .mobilenet import MobileNet
from .squeeze_net import SqueezeNet
from .resnet import ResNet
from .shuffle_net import ShuffleNet

__all__ = [
    'SimpleNet', 'EfficientNet', 'MobileNet', 
    'SqueezeNet', 'ResNet', 'ShuffleNet'
]