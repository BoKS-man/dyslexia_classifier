import torchvision.models as models
from torch import nn
import torch

class DysClassifier(nn.Module):
    def __init__(self):
        super(DysClassifier, self).__init__()
        self.__backbone = models.efficientnet_b0(weights=models.efficientnet.EfficientNet_B0_Weights.DEFAULT)
        self.__backbone.eval()
        self.__do_0 = nn.Dropout(p=0.4)
        self.__ln_0 = nn.Linear(1000,200)
        self.__do_1 = nn.Dropout(p=0.2)
        self.__ln_1 = nn.Linear(200,1)

    def forward(self, x):
        x = self.__backbone(x)
        x = self.__do_0(x)
        x = self.__ln_0(x)
        x = self.__do_1(x)
        x = self.__ln_1(x)
        x = torch.sigmoid(x)
        return x