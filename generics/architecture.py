from generics.data import Data
from pytorch.network import PytorchNetwork


class Architecture:
    def __init__(self, data: Data, name: str = None):
        self.data = data
        self.name = name

    def build(self):
        layers = []
        for section in self.data.sections:
            layers += section.get_layers()
        return PytorchNetwork(layers)
