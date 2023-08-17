from block import Block
from layer import Layer
from model import Model
from section import Section


class Architecture:
    def __init__(self, network: Model, name: str = None):
        self.structure = []
        self.network = network
        self.name = name

    def attach(self, component: Section | Block | Layer):
        self.structure.append(component.get()) if isinstance(component, Layer) else self.structure.extend(component.layers())
        return self

    def build(self):
        return self.network.build(self.structure)
