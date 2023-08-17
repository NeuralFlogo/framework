from torch import nn
from layers.normalization import NormalizationLayer


class PytorchBatchNormalization(NormalizationLayer):
    def __init__(self, out_channels, probability=0.1, eps=1e-5):
        self.layer = nn.BatchNorm2d(out_channels, eps=eps, momentum=probability)

    def get(self):
        return self.layer
