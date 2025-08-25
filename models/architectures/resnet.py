import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=10, arch="resnet18", pretrained=True):
        super(ResNet, self).__init__()
        if '18' in arch:
            archi = models.resnet18
        elif '34' in arch:
            archi = models.resnet34
        elif '50' in arch:
            archi = models.resnet50
        elif '101' in arch:
            archi = models.resnet101
        elif '152' in arch:
            archi = models.resnet152
        else:
            raise NotImplementedError("The architecture is not implemented")

        if pretrained:
            # DEFAULT means the best available weights from ImageNet.
            self.model = archi(weights="DEFAULT")
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes, bias=True)

        else:
            self.model = archi(weights=None, num_classes=num_classes)

    def forward(self, x, return_features=False):
        if return_features:
            # Get features before the final classifier
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            features = x.view(x.size(0), -1)
            out = self.model.fc(features)
            return out, features
        else:
            return self.model(x)