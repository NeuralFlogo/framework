from torch import Tensor

from framework.architecture.layers.slicing import SlicingLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchSlicingLayer(PytorchLayer, SlicingLayer):
    def __init__(self, start: int = 0, end: int = None):
        PytorchLayer.__init__(self)
        SlicingLayer.__init__(self, start, end)

    def forward(self, x: Tensor) -> Tensor:
        return x[:, self.start:self.end, :]
