from torch import nn

from layers.convolutional import ConvolutionalLayer


class PytorchConvolutional(ConvolutionalLayer):

    def __init__(self, in_channels: int, out_channels: int, kernel=3, stride=1, padding=0):
        super().__init__(kernel, in_channels, out_channels, stride, padding)
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.Conv2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                         in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, x):
        return self.layer(x)
