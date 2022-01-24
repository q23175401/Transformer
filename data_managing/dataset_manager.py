import numpy as np
import torch as th
import torch.nn as nn

from data_preprocessor import DataPreprocessor


class DatasetManager:
    def __init__(self, dataset_path) -> None:
        self.processor = DataPreprocessor()

    def _load(self, dataset_path):
        pass

    def getBatchData(self, batch_size):
        return

    def getDataPyIndex(self, index):
        return

