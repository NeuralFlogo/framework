from torch import nn

from layers.convolutional import ConvolutionalLayer


class PytorchConvolutional(ConvolutionalLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel=3, stride=1, padding=0):
        self.layer = nn.Conv2d(kernel_size=kernel, stride=stride, padding=padding, in_channels=in_channels, out_channels=out_channels)

    def get(self):
        return self.layer
