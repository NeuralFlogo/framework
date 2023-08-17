from torch import nn
from layers.pool import PoolingLayer


class PytorchMaxPooling(PoolingLayer):
    def __init__(self, kernel=2, stride=2, padding=0):
        self.layer = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def get(self):
        return self.layer
