from layer import Layer


class NormalizationLayer(Layer):
    def __init__(self, probability: float):
        self.probability = probability
