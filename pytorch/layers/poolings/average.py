from torch import nn
from layers.pool import PoolingLayer


class PytorchAveragePooling(PoolingLayer):
    def __init__(self, kernel, stride, padding):
        self.layer = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def get(self):
        return self.layer
