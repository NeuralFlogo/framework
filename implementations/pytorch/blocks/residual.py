from typing import List, Union

from implementations.pytorch.block import PytorchBlock
from implementations.pytorch.layers.activation import PytorchActivationLayer
from implementations.pytorch.layers.convolutional import PytorchConvolutionalLayer
from implementations.pytorch.layers.regularization import PytorchRegularizationLayer


class PytorchResidualBlock(PytorchBlock):
    def __init__(self, layers: List[Union[PytorchConvolutionalLayer, PytorchRegularizationLayer, PytorchActivationLayer]]):
        super(PytorchResidualBlock, self).__init__(layers)
