import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
    def __init__(self, name:str, num_classes:int):
        super(Classifier, self).__init__()

        if name == 'resnet101':
            self.encoder = models.resnet101(pretrained=True)

        if name == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)

        if name == 'efficientnet-b3':
            self.encoder = EfficientNet.from_pretrained(name)

        if name == 'inceptionv3':
            self.encoder = models.inception_v3(pretrained=True)
            num_ftrs = self.encoder.AuxLogits.fc.in_features
            self.encoder.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)


        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes, bias=True)
        )

    def forward(self, images):
        return self.encoder(images)