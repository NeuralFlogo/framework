from framework.architecture.layers.regularization import RegularizationLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchRegularizationLayer(PytorchLayer, RegularizationLayer):
    def __init__(self):
        super(PytorchRegularizationLayer, self).__init__()
