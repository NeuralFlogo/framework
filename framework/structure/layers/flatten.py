from layer import Layer


class FlattenLayer(Layer):
    def __init__(self, start_dim: int, end_dim: int):
        self.start_dim = start_dim
        self.end_dim = end_dim


