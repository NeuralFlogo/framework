from typing import List, Union

from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer
from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchLinearBlock(PytorchBlock):
    def __init__(self, layers: List[Union[PytorchLinearLayer, PytorchRegularizationLayer, PytorchActivationLayer]]):
        super(PytorchLinearBlock, self).__init__(layers)
