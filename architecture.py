from input import Input
from layer import Layer


class Architecture:
    def __init__(self, inputs: Input, network, name: str = None):  #TODO allow multiple inputs (list of inputs) for Resnet
        self.inputs = inputs
        self.name = name
        self.network = network

    def build(self):
        layers = []
        for step in self.inputs.trail:
            layers.append(step) if isinstance(step, Layer) else layers += step.layers()
        return self.network.build(layers)
