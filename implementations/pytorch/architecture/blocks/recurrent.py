from typing import List, Union

from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer
from implementations.pytorch.architecture.layers.recurrent import PytorchRecurrentLayer
from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchRecurrentBlock(PytorchBlock):
    def __init__(self, layers: List[Union[PytorchRecurrentLayer, PytorchRegularizationLayer, PytorchActivationLayer]]):
        super(PytorchRecurrentBlock, self).__init__(layers)
