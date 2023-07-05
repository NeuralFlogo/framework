from torch import nn

from layers.normalization import NormalizationLayer


class PytorchDropout(NormalizationLayer):

    def __init__(self, probability):
        super().__init__(probability)
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.Dropout(p=self.probability)

    def forward(self, x):
        return self.layer(x)
