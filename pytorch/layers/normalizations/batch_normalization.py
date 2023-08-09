from torch import nn
from framework.structure.layers.normalization import NormalizationLayer


class PytorchBatchNormalization(nn.Module, NormalizationLayer):
    def __init__(self, out_channels, probability=0.1, eps=1e-5):
        nn.Module.__init__(self)
        NormalizationLayer.__init__(self, probability)
        self.layer = nn.BatchNorm2d(out_channels, eps=eps, momentum=self.probability)

    def forward(self, x):
        return self.layer(x)
