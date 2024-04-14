from typing import Tuple

from torch import Tensor

from framework.architecture.layers.slicing import SlicingLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchSlicingLayer(PytorchLayer, SlicingLayer):
    def __init__(self, output: SlicingLayer.OutputType, start: int = 0, end: int = None):
        PytorchLayer.__init__(self)
        SlicingLayer.__init__(self, output, start, end)

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        return x[self.output.value][self.start:self.end]
