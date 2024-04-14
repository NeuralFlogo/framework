from torch import Tensor
from torch.nn import Module

from framework.architecture.layer import Layer


class PytorchLayer(Module, Layer):
    def __init__(self):
        Module.__init__(self)
        Layer.__init__(self)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
