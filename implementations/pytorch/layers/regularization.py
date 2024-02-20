from framework.layers.regularization import RegularizationLayer
from implementations.pytorch.layer import PytorchLayer


class PytorchRegularizationLayer(PytorchLayer, RegularizationLayer):
    def __init__(self):
        super(PytorchRegularizationLayer, self).__init__()
