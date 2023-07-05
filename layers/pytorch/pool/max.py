from torch import nn

from layers.pool import PoolLayer


class PytorchMaxPooling(PoolLayer):

    def __init__(self, kernel=2, stride=2, padding=0):
        super().__init__(kernel, stride, padding)
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)

    def forward(self, x):
        return self.layer(x)
