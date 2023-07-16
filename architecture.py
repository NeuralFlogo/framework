from input import Input
from pytorch.network import PytorchNetwork


class Architecture:
    def __init__(self, inputs: Input, name: str = None):
        self.inputs = inputs
        self.name = name

    def build(self):
        layers = []
        for section in self.data.sections:
            layers += section.get_layers()
        return PytorchNetwork(layers)
