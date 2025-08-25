import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url
import torchvision.models as models

class EfficientNet(nn.Module):
    """
    A CNN model based on EfficientNet (B0 or B4)
    """
    def __init__(self, num_classes=10, arch="efficientnetB4", pretrained=True):
        super(EfficientNet, self).__init__()

        # ////////////////////////////////// efficientNet (B0 ou B4) ///////////////////////////////////////
        if '0' in arch:
            # problem with the actual version of efficientnet (B0)  weights in PyTorch Hub
            # when this problem will be solved, we can use directly the following line
            # archi = models.efficientnet_b0 if '0' in arch else models.efficientnet_b4
            def get_state_dict(self, *args, **kwargs):
                kwargs.pop("check_hash")
                return load_state_dict_from_url(self.url, *args, **kwargs)

            models._api.WeightsEnum.get_state_dict = get_state_dict

            archi = models.efficientnet_b0

        elif '4' in arch:
            archi = models.efficientnet_b4

        else:
            raise NotImplementedError("The architecture is not implemented")

        if pretrained:
            self.model = archi(weights="DEFAULT")  # DEFAULT means the best available weights from ImageNet.
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            # Note: the softmax function is not used here because it is included in the loss function

        else:
            self.model = archi(weights=None, num_classes=num_classes)

    def forward(self, x, return_features=False):
        out = self.model.features(x)
        out = self.model.avgpool(out)

        features = out.view(out.size(0), -1)

        out = self.model.classifier(features)

        return (out, features) if return_features else out