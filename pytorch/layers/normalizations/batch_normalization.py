from torch import nn
from layers.normalization import NormalizationLayer


class PytorchBatchNormalization(NormalizationLayer):
    def __init__(self, out_channels, probability=0.1, eps=1e-5):
        super().__init__(probability)
        self.layer = nn.BatchNorm2d(out_channels, eps=eps, momentum=self.probability)

    def forward(self, x):
        return self.layer(x)
