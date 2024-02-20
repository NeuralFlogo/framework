from framework.layers.activation import ActivationLayer
from implementations.pytorch.layer import PytorchLayer


class PytorchActivationLayer(PytorchLayer, ActivationLayer):
    def __init__(self):
        super(PytorchActivationLayer, self).__init__()

