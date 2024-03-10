from torch import Tensor
from torch.nn import Sequential

from framework.architecture.layers.residual import ResidualLayer
from implementations.pytorch.architecture.layer import PytorchLayer
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer
from implementations.pytorch.architecture.layers.convolutional import PytorchConvolutionalLayer
from implementations.pytorch.architecture.layers.regularizations.batch_normalization import \
    PytorchBidimensionalBatchNormalizationLayer


class PytorchResidualLayer(PytorchLayer, ResidualLayer):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(PytorchResidualLayer, self).__init__()
        self.conv1 = PytorchConvolutionalLayer(in_channels, out_channels, kernel=3, stride=stride, padding=1)
        self.bn1 = PytorchBidimensionalBatchNormalizationLayer(out_channels)
        self.relu = PytorchReLULayer(inplace=True)
        self.conv2 = PytorchConvolutionalLayer(out_channels, out_channels, kernel=3, stride=1, padding=1)
        self.bn2 = PytorchBidimensionalBatchNormalizationLayer(out_channels)
        self.shortcut = self.__build_skip_connection(in_channels, out_channels, stride)

    def __build_skip_connection(self, in_channels: int, out_channels: int, stride: int):
        if stride != 1 or in_channels != out_channels:
            return Sequential(
                PytorchConvolutionalLayer(in_channels, out_channels, kernel=1, stride=stride),
                PytorchBidimensionalBatchNormalizationLayer(out_channels))
        return None

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
