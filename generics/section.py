
class Section:

    def __init__(self, blocks):
        self.blocks = blocks

    def get_layers(self):
        layers = []
        for block in self.blocks:
            layers += block.get_layers()
        return layers

    def __call__(self, x):
        x.add(self)
        return x
