from torch import nn
from layers.pool import PoolingLayer


class PytorchAveragePooling(nn.Module, PoolingLayer):
    def __init__(self, kernel, stride, padding):
        nn.Module.__init__(self)
        PoolingLayer.__init__(self, kernel, stride, padding)
        self.layer = nn.AvgPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)

    def forward(self, x):
        return self.layer(x)

