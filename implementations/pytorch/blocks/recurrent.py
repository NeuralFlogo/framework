from typing import List, Union

from implementations.pytorch.block import PytorchBlock
from implementations.pytorch.layers.activation import PytorchActivationLayer
from implementations.pytorch.layers.recurrent import PytorchRecurrentLayer
from implementations.pytorch.layers.regularization import PytorchRegularizationLayer


class PytorchRecurrentBlock(PytorchBlock):
    def __init__(self, layers: List[Union[PytorchRecurrentLayer, PytorchRegularizationLayer, PytorchActivationLayer]]):
        super(PytorchRecurrentBlock, self).__init__(layers)
