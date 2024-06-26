from typing import List

from framework.architecture.layer import Layer


class Block:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
