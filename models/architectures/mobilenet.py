import torch.nn as nn
import torchvision.models as models

class MobileNet(nn.Module):
    """
    A CNN model based on MobileNet (V2)
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileNet, self).__init__()
        if pretrained:
            # DEFAULT means the best available weights from ImageNet.
            self.model = models.mobilenet_v2(weights='DEFAULT')
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            # Note: the softmax function is not used here because it is included in the loss function

        else:
            self.model = models.mobilenet_v2(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)