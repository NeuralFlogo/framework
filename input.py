class Track:
    def __init__(self, shape):
        self.shape = shape
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return self
