from typing import Union, Tuple

from torch import nn

from implementations.pytorch.layers.pool import PytorchPoolingLayer


class PytorchMaxPoolingLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(PytorchMaxPoolingLayer, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
