from torch import nn
from layers.normalization import NormalizationLayer


class PytorchDropout(NormalizationLayer):
    def __init__(self, probability):
        self.layer = nn.Dropout(p=probability)

    def get(self):
        return self.layer
