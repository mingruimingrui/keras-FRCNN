"""This file should contain the interface for the user to build their model with"""

"""Computer vision"""

# General CV algorithms
from . import backbone
from .inception import InceptionV3, InceptionV3Backbone
from .resnet import ResNet50, ResNet50Backbone
from .vgg import VGG16, VGG16Backbone

# Detection
from .retinanet_config import RetinaNetConfig
from .retinanet import (
    RetinaNet,
    RetinaNetTrain,
    RetinaNetFromTrain,
    LoadRetinaNet
)

# Facial Recognition

# Feature Extraction

# GANs
