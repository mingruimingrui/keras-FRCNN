"""This file should contain the interface for the user to build their model with"""

"""Computer vision"""
# General CV algorithms
from .inception import InceptionV3

# Detection
from .retinanet import RetinaNet, RetinaNetTrain
from ._config import RetinaNetConfig


# def load_model_config_template():
#     return ModelConfigTemplate()
#
#
# def load_model(config):
#     config.check_valid()
#     return RetinaNet(config)
