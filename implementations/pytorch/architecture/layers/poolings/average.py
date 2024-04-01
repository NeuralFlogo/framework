from typing import Union, Tuple

from torch import nn

from implementations.pytorch.architecture.layers.pool import PytorchPoolingLayer


class PytorchAvgPoolLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = None, padding: Union[int, Tuple[int, int]] = 0):
        super(PytorchAvgPoolLayer, self).__init__()
        self.layer = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)
