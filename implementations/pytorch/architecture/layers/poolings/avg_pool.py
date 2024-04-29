from typing import Union, Tuple

from torch import nn

from implementations.pytorch.architecture.layers.pool import PytorchPoolingLayer


class Pytorch1DimensionalAvgPoolLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = None, padding: Union[int, Tuple[int, int]] = 0):
        super(Pytorch1DimensionalAvgPoolLayer, self).__init__()
        self.layer = nn.AvgPool1d(kernel_size=kernel, stride=stride, padding=padding)


class Pytorch2DimensionalAvgPoolLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = None, padding: Union[int, Tuple[int, int]] = 0):
        super(Pytorch2DimensionalAvgPoolLayer, self).__init__()
        self.layer = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)


class Pytorch3DimensionalAvgPoolLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = None, padding: Union[int, Tuple[int, int]] = 0):
        super(Pytorch3DimensionalAvgPoolLayer, self).__init__()
        self.layer = nn.AvgPool3d(kernel_size=kernel, stride=stride, padding=padding)