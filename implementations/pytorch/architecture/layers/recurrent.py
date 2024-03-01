from framework.architecture.layers.recurrent import RecurrentLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchRecurrentLayer(PytorchLayer, RecurrentLayer):
    def __init__(self):
        super(PytorchRecurrentLayer, self).__init__()
