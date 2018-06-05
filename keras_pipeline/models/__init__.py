"""This file should contain the interface for the user to build their model with"""

"""Computer vision"""

# General CV algorithms
from .inception import InceptionV3
from .resnet import ResNet50
from .vgg import VGG16
from .load_backbone import load_backbone

# Detection
from .retinanet import RetinaNet, RetinaNetTrain, RetinaNetFromTrain
from .model_config import RetinaNetConfig

# Feature Extraction

# GANs
