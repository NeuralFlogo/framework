from typing import List, Union

from framework.architecture.block import Block
from framework.architecture.layers.activation import ActivationLayer
from framework.architecture.layers.convolutional import ConvolutionalLayer
from framework.architecture.layers.regularization import RegularizationLayer


class ConvolutionalBlock(Block):
    def __init__(self, layers: List[Union[ConvolutionalLayer, RegularizationLayer, ActivationLayer]]):
        super().__init__(layers)
