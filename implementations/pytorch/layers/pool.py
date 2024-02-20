from framework.layers.pool import PoolingLayer
from implementations.pytorch.layer import PytorchLayer


class PytorchPoolingLayer(PytorchLayer, PoolingLayer):
    def __init__(self):
        super(PytorchPoolingLayer, self).__init__()
