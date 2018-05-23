"""This file should contain the interface for the user to build their model with"""

from ._config import ModelConfigTemplate
from .frcnn import FRCNN


def load_model_config_template():
    return ModelConfigTemplate()


def load_model(config):
    config.check_valid()
    return FRCNN(config)
