import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

from src.wider_dataset import WiderDataset

wider = WiderDataset('D://datasets/wider', 'val')

# print(wider.image_infos[wider.image_infos.keys()[0]])
# print(wider.image_infos.keys())
cur_id = list(wider.image_infos.keys())[0]
print(wider.image_infos[cur_id])
