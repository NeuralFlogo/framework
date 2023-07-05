from torch import nn

from layers.normalization import NormalizationLayer


class PytorchBatchNormalization(NormalizationLayer):

    def __init__(self, out_channels, probability=0.1, eps=1e-5):
        super().__init__(probability)
        self.out_channels = out_channels
        self.eps = eps
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.BatchNorm2d(self.out_channels, eps=self.eps, momentum=self.probability)

    def forward(self, x):
        return self.layer(x)
