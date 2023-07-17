from layer import Layer


class LinearLayer(Layer):
    def __init__(self, in_dimension, out_dimension):
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
