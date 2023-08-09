from torch import nn
from framework.structure.layers.normalization import NormalizationLayer


class PytorchDropout(nn.Module, NormalizationLayer):
    def __init__(self, probability):
        nn.Module.__init__(self)
        NormalizationLayer.__init__(self, probability)
        self.layer = nn.Dropout(p=self.probability)

    def forward(self, x):
        return self.layer(x)
