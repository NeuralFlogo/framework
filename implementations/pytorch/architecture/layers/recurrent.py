from typing import Tuple

from torch import Tensor

from framework.architecture.layers.recurrent import RecurrentLayer, SlicingLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchRecurrentLayer(PytorchLayer, RecurrentLayer):
    def __init__(self):
        super(PytorchRecurrentLayer, self).__init__()


class PytorchSlicingLayer(PytorchLayer, SlicingLayer):
    def __init__(self, output: SlicingLayer.OutputType, start: int = 0, end: int = None):
        PytorchLayer.__init__(self)
        SlicingLayer.__init__(self, output, start, end)

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        return x[self.output.value][self.start:self.end]