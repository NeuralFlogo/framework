from framework.architecture.layers.residual import ResidualLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchResidualLayer(PytorchLayer, ResidualLayer):
    def __init__(self):
        super(PytorchResidualLayer, self).__init__()
