from torch import Tensor

from framework.architecture.layers.recurrent import RecurrentLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchRecurrentLayer(PytorchLayer, RecurrentLayer):
    def __init__(self, output_type: RecurrentLayer.OutputType):
        PytorchLayer.__init__(self)
        RecurrentLayer.__init__(self, output_type)

    def forward(self, x: Tensor) -> Tensor:
        end_sequence, hidden = self.layer(x)
        return (end_sequence, hidden.transpose(0, 1))[self.output.value]
