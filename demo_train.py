import os
import sys

from keras_pipeline.models import RetinaNet, RetinaNetTrain, RetinaNetConfig

# Configure model configs
model_config = RetinaNetConfig(num_classes = 7)

model = RetinaNet(model_config)
