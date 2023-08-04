import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class FeatureVGG19(nn.Module):
    def __init__(self):
        super(FeatureVGG19, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        for param in vgg19_54.parameters():
            param.requires_grad = False
        self.vgg19_54 = vgg19_54

    def forward(self, img):
        return self.vgg19_54(img)
