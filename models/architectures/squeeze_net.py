import torch.nn as nn
import torch
import torchvision.models as models

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(SqueezeNet, self).__init__()
        if pretrained:
            # DEFAULT means the best available weights from ImageNet.
            self.model = models.squeezenet1_0(weights='DEFAULT')

            num_ftrs = self.model.classifier[1].in_channels
            self.model.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1))
            # Note: the softmax function is not used here because it is included in the loss function

        else:
            self.model = models.squeezenet1_0(weights=None, num_classes=num_classes)

    def forward(self, x, return_features=False):
        out = self.model.features(x)
        out = self.model.classifier(out)  # self.model.classifier(features)
        out = torch.flatten(out, 1)

        if return_features:
            features = out.view(out.size(0), -1)
            return out, features

        return out