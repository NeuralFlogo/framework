from torch import nn
from layers.pool import PoolingLayer


class PytorchAveragePooling(PoolingLayer):
    def __init__(self, kernel, stride, padding):
        super().__init__(kernel, stride, padding)
        self.layer = nn.AvgPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)

    def forward(self, x):
        return self.layer(x)

