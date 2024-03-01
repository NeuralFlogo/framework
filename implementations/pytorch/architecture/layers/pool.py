from framework.architecture.layers.pool import PoolingLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchPoolingLayer(PytorchLayer, PoolingLayer):
    def __init__(self):
        super(PytorchPoolingLayer, self).__init__()
