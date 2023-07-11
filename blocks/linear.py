from layers.activation import ActivationLayer
from layers.linear import LinearLayer
from layers.normalization import NormalizationLayer


class LinearBlock:
    def __init__(self, linear: LinearLayer, activation: ActivationLayer = None, normalization: NormalizationLayer = None):
        self.linear = linear
        self.activation = activation
        self.normalization = normalization
