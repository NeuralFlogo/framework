from typing import List, Union

from framework.block import Block
from framework.layers.activation import ActivationLayer
from framework.layers.convolutional import ConvolutionalLayer
from framework.layers.regularization import RegularizationLayer


class ResidualBlock(Block):
    def __init__(self, layers: List[Union[ConvolutionalLayer, RegularizationLayer, ActivationLayer]]):
        super().__init__(layers)
