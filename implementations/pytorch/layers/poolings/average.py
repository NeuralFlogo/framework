from typing import Union, Tuple

from torch import nn

from implementations.pytorch.layers.pool import PytorchPoolingLayer


class PytorchAveragePoolingLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(PytorchAveragePoolingLayer, self).__init__()
        self.layer = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)
