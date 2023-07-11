from torch import nn
from layers.pool import PoolingLayer


class PytorchMaxPooling(PoolingLayer):
    def __init__(self, kernel=2, stride=2, padding=0):
        super().__init__(kernel, stride, padding)
        self.layer = nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)

    def forward(self, x):
        return self.layer(x)
