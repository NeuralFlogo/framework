from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU

from layers.residual import ResidualLayer


class PytorchResidual(ResidualLayer):
    def __init__(self, in_features, out_features: int, hidden_size: int, expansion: int, stride: int = 1):
        self.layer = Sequential(*tuple(self.__create_residual_block(i, in_features, out_features, expansion, stride) for i in range(hidden_size)))

    def __create_residual_block(self, idx, in_features, out_features, expansion, stride):
        if idx == 0:
            return self.ResidualBlock(in_features, out_features, stride)
        return self.ResidualBlock(in_features * expansion, out_features, expansion)

    def get(self):
        return self.layer

    class ResidualBlock(Module):
        def __init__(self, in_features, out_features, expansion, stride=1):
            super().__init__()
            self.conv1 = Sequential(
                Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0),
                BatchNorm2d(out_features))
            self.conv2 = Sequential(
                Conv2d(out_features, out_features, kernel_size=3, stride=stride, padding=1),
                BatchNorm2d(out_features))
            self.conv3 = Sequential(
                Conv2d(out_features, out_features * 4, kernel_size=1, stride=1, padding=0),
                BatchNorm2d(out_features * 4))
            self.activation = ReLU()
            self.downsample = self.__build_downsample(in_features, out_features, expansion, stride)

        def __build_downsample(self, in_features, out_features, expansion, stride):
            return self.__create_conv_block(in_features, out_features, expansion, stride)\
                if self.__is_size_different(in_features, out_features, expansion, stride) else None

        def __is_size_different(self, in_features, out_features, expansion, stride):
            return stride != 1 or in_features != out_features * expansion

        def __create_conv_block(self, in_features, out_features, expansion, stride):
            return Sequential(
                    Conv2d(in_features, out_features * expansion, kernel_size=1, stride=stride),
                    BatchNorm2d(out_features * expansion))

        def forward(self, x):
            residue = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            if self.downsample:
                residue = self.downsample(residue)
            return self.activation(x + residue)
