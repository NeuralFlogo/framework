from typing import Union, Tuple

from torch import nn

from implementations.pytorch.architecture.layers.pool import PytorchPoolingLayer


class Pytorch1DimensionalMaxPoolLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(Pytorch1DimensionalMaxPoolLayer, self).__init__()
        self.layer = nn.MaxPool1d(kernel_size=kernel, stride=stride, padding=padding)


class Pytorch2DimensionalMaxPoolLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(Pytorch2DimensionalMaxPoolLayer, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)


class Pytorch3DimensionalMaxPoolLayer(PytorchPoolingLayer):
    def __init__(self, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(Pytorch3DimensionalMaxPoolLayer, self).__init__()
        self.layer = nn.MaxPool3d(kernel_size=kernel, stride=stride, padding=padding)
