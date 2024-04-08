from typing import Tuple

from torch import Tensor

from framework.architecture.layers.slicer import SlicingLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchSlicingLayer(PytorchLayer, SlicingLayer):
    def __init__(self, output: SlicingLayer.OutputType, start: int, end: int):
        super(PytorchSlicingLayer, self).__init__(output, start, end)

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        return x[self.output.value][self.start:self.end, :, :]