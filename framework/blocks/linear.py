from typing import List, Union

from framework.block import Block
from framework.layers.activation import ActivationLayer
from framework.layers.linear import LinearLayer
from framework.layers.regularization import RegularizationLayer


class LinearBlock(Block):
    def __init__(self, layers: List[Union[LinearLayer, RegularizationLayer, ActivationLayer]]):
        super().__init__(layers)
