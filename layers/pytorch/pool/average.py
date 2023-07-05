from torch import nn

from layers.pool import PoolLayer


class PytorchAveragePooling(PoolLayer):

    def __init__(self, kernel, stride, padding):
        super().__init__(kernel, stride, padding)
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.AvgPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)

    def forward(self, x):
        return self.layer(x)

