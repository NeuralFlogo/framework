from framework.architecture.layers.activation import ActivationLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchActivationLayer(PytorchLayer, ActivationLayer):
    def __init__(self):
        super(PytorchActivationLayer, self).__init__()

