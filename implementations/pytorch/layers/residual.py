from framework.layers.residual import ResidualLayer
from implementations.pytorch.layer import PytorchLayer


class PytorchResidualLayer(PytorchLayer, ResidualLayer):
    def __init__(self):
        super(PytorchResidualLayer, self).__init__()
