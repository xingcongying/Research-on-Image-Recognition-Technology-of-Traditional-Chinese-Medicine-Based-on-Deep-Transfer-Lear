import torchvision
import torch.nn.functional as F 
from torch import nn
from config import config


def get_net():
    #return MyModel(torchvision.models.resnet101(pretrained = True))
    model = torchvision.models.densenet121(pretrained = True)
    #model = torchvision.models.resnet152(pretrained = True)
    #for param in model.parameters():
    #    param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048,config.num_classes)
    return model

