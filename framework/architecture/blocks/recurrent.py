from typing import List, Union

from framework.architecture.block import Block
from framework.architecture.layers.activation import ActivationLayer
from framework.architecture.layers.regularization import RegularizationLayer
from framework.architecture.layers.recurrent import RecurrentLayer


class RecurrentBlock(Block):
    def __init__(self, layers: List[Union[RecurrentLayer, RegularizationLayer, ActivationLayer]]):
        super().__init__(layers)
