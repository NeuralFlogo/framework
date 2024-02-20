from typing import List, Union

from implementations.pytorch.block import PytorchBlock
from implementations.pytorch.layers.activation import PytorchActivationLayer
from implementations.pytorch.layers.linear import PytorchLinearLayer
from implementations.pytorch.layers.regularization import PytorchRegularizationLayer


class PytorchLinearBlock(PytorchBlock):
    def __init__(self, layers: List[Union[PytorchLinearLayer, PytorchRegularizationLayer, PytorchActivationLayer]]):
        super(PytorchLinearBlock, self).__init__(layers)
