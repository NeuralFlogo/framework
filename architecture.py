from input import Input
from layer import Layer


class Architecture:
    def __init__(self, input: Input, network, name: str = None):
        self.input = input
        self.network = network
        self.name = name
        self.layers = []

    def build(self):
        for step in self.input.route:
            if isinstance(step, Layer):
                self.layers.append(step)
            else:
                self.layers += step.layers()
        return self.network.build(self.layers)
