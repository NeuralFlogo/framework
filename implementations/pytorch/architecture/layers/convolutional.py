from typing import Union, Tuple

from torch import nn

from framework.architecture.layers.convolutional import ConvolutionalLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class Pytorch1DimensionalConvolutionalLayer(PytorchLayer, ConvolutionalLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(Pytorch1DimensionalConvolutionalLayer, self).__init__()
        self.layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)


class Pytorch2DimensionalConvolutionalLayer(PytorchLayer, ConvolutionalLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(Pytorch2DimensionalConvolutionalLayer, self).__init__()
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)


class Pytorch3DimensionalConvolutionalLayer(PytorchLayer, ConvolutionalLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        super(Pytorch3DimensionalConvolutionalLayer, self).__init__()
        self.layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)
