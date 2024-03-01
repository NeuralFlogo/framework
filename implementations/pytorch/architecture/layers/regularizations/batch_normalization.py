from torch import nn

from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchBatchNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, out_channels: int, probability: float, eps: float):
        super(PytorchBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm2d(out_channels, eps=eps, momentum=probability)
