import torch.nn as nn
import torchvision.models as models
import ssl

class ShuffleNet(nn.Module):
    def __init__(self, num_classes=10, size='1.0x', pretrained=True):
        super(ShuffleNet, self).__init__()
        
        # Disable SSL verification for model downloads if needed
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception as e:
            print(f"Warning: Could not modify SSL context: {e}")
        
        # Select architecture based on size parameter
        if size == '0.5x':
            archi = models.shufflenet_v2_x0_5
        elif size == '1.0x':
            archi = models.shufflenet_v2_x1_0
        elif size == '1.5x':
            archi = models.shufflenet_v2_x1_5
        elif size == '2.0x':
            archi = models.shufflenet_v2_x2_0
        else:
            raise NotImplementedError(f"ShuffleNet size {size} not implemented. Choose from: '0.5x', '1.0x', '1.5x', '2.0x'")

        try:
            if pretrained:
                print("Attempting to load pretrained weights...")
                self.model = archi(weights='DEFAULT')
                print("Successfully loaded pretrained weights")
                # Modify the classifier for our number of classes
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            else:
                print("Initializing model without pretrained weights")
                self.model = archi(weights=None, num_classes=num_classes)
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
            print("Falling back to random initialization")
            self.model = archi(weights=None, num_classes=num_classes)

    def forward(self, x, return_features=False):
        if return_features:
            # Get features before the final classifier
            features = self.model.conv1(x)
            features = self.model.maxpool(features)
            features = self.model.stage2(features)
            features = self.model.stage3(features)
            features = self.model.stage4(features)
            features = self.model.conv5(features)
            features = features.mean([2, 3])  # Global average pooling
            out = self.model.fc(features)
            return out, features
        else:
            return self.model(x)