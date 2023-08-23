from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential

from layers.residual import ResidualLayer

BASE_INPUT = 64


class PytorchResidual(ResidualLayer):
    def __init__(self, in_features: int, stride: int, hidden_size: int, expansion: int = 4):
        layers = []
        for i in range(hidden_size):
            if i == 0:
                layers.append(self.Bottleneck(in_features=self.__in_feat(in_features), out_features=in_features, expansion=expansion, stride=stride))
            else:
                layers.append(self.Bottleneck(in_features=in_features * expansion, out_features=in_features, expansion=expansion, stride=1))
        self.layer = Sequential(*layers)

    def get(self):
        return self.layer

    @staticmethod
    def __in_feat(in_features):
        if in_features == BASE_INPUT: return in_features
        return in_features * 2

    class Bottleneck(Module):
        def __init__(self, in_features: int, out_features: int, expansion: int,  stride: int):
            super().__init__()
            self.expansion = expansion
            self.conv1 = Conv2d(in_features, out_features, kernel_size=1, stride=1)
            self.bn1 = BatchNorm2d(out_features)
            self.conv2 = Conv2d(out_features, out_features, kernel_size=3, stride=stride)
            self.bn2 = BatchNorm2d(out_features)
            self.conv3 = Conv2d(out_features, out_features * self.expansion, kernel_size=1, stride=1)
            self.bn3 = BatchNorm2d(out_features * self.expansion)
            self.relu = ReLU()
            self.downsample = self.__init_downsample(in_features, out_features, stride)

        def __init_downsample(self, in_features, out_features, stride):
            if stride != 1 or in_features != out_features * self.expansion:
                return self.__build_conv_block(in_features, out_features, stride)
            else:
                return None

        def __build_conv_block(self, in_features, out_features, stride):
            return Sequential(
                Conv2d(in_features, out_features * self.expansion, kernel_size=1, stride=stride),
                BatchNorm2d(out_features * self.expansion))

        def forward(self, x):
            residue = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            if self.downsample:
                residue = self.downsample(residue)
            return self.relu(x + residue)
