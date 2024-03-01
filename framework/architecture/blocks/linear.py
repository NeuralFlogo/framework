from typing import List, Union

from framework.architecture.block import Block
from framework.architecture.layers.activation import ActivationLayer
from framework.architecture.layers.linear import LinearLayer
from framework.architecture.layers.regularization import RegularizationLayer


class LinearBlock(Block):
    def __init__(self, layers: List[Union[LinearLayer, RegularizationLayer, ActivationLayer]]):
        super().__init__(layers)
