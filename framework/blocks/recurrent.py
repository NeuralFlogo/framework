from typing import List, Union

from framework.block import Block
from framework.layers.activation import ActivationLayer
from framework.layers.regularization import RegularizationLayer
from framework.layers.recurrent import RecurrentLayer


class RecurrentBlock(Block):
    def __init__(self, layers: List[Union[RecurrentLayer, RegularizationLayer, ActivationLayer]]):
        super().__init__(layers)
