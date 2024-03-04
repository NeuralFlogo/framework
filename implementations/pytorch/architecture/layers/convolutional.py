from torch import nn

from framework.architecture.layers.convolutional import ConvolutionalLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchConvolutionalLayer(PytorchLayer, ConvolutionalLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, padding: int = 0):
        super(PytorchConvolutionalLayer, self).__init__()
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)
