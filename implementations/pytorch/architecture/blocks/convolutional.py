from typing import List, Union

from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer
from implementations.pytorch.architecture.layers.convolutional import PytorchConvolutionalLayer
from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchConvolutionalBlock(PytorchBlock):
    def __init__(self, layers: List[Union[PytorchConvolutionalLayer, PytorchRegularizationLayer, PytorchActivationLayer]]):
        super(PytorchConvolutionalBlock, self).__init__(layers)
