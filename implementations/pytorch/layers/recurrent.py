from framework.layers.recurrent import RecurrentLayer
from implementations.pytorch.layer import PytorchLayer


class PytorchRecurrentLayer(PytorchLayer, RecurrentLayer):
    def __init__(self):
        super(PytorchRecurrentLayer, self).__init__()
