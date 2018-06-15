"""This file should contain the interface for the user to build their model with"""

"""Computer vision"""

# General CV algorithms
from .inception import InceptionV3, InceptionV3Backbone
from .resnet import ResNet50, ResNet50Backbone
from .vgg import VGG16, VGG16Backbone

# Detection
from .model_config import RetinaNetConfig
from .retinanet import RetinaNet
from .retinanet import RetinaNetTrain, RetinaNetFromTrain
from .retinanet import LoadRetinaNet

# Feature Extraction

# GANs
