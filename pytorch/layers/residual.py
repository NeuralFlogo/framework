from torch.nn import Module, Sequential, ReLU, Conv2d, BatchNorm2d

from layers.residual import ResidualLayer

BASE_INPUT = 64


class PytorchResidual(ResidualLayer):
    def __init__(self, in_features: int, hidden_size: int, expansion: int = 4):
        self.in_features = in_features
        self.expansion = expansion
        self.layer = Sequential(*tuple(self.__build_block(i) for i in range(hidden_size)))

    def get(self):
        return self.layer

    def __build_block(self, idx):
        if idx == 0:
            return self.Block(in_features=self.__match_input_size(), out_features=self.in_features,
                              expansion=self.expansion, stride=2)
        return self.Block(in_features=self.in_features * self.expansion, out_features=self.in_features,
                          expansion=self.expansion)

    def __match_input_size(self):
        if self.in_features == BASE_INPUT:
            return self.in_features
        return self.in_features // 2 if self.expansion == 1 else self.in_features * 2

    class Block(Module):
        def __init__(self, in_features: int, out_features: int, expansion: int, stride: int = 1):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.expansion = expansion
            self.stride = self.__compute_stride(stride)
            self.convolution = self.__build_basic_conv() if expansion == 1 else self.__build_bottleneck_conv()
            self.activation = ReLU()
            self.downsample = self.__init_downsample()

        def __compute_stride(self, stride):
            return 1 if self.in_features == BASE_INPUT and self.in_features == self.out_features else stride

        def __build_basic_conv(self):
            return Sequential(
                Conv2d(self.in_features, self.out_features, kernel_size=3, stride=self.stride, padding=1),
                BatchNorm2d(self.out_features),
                ReLU(),
                Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(self.out_features))

        def __build_bottleneck_conv(self):
            return Sequential(
                Conv2d(self.in_features, self.out_features, kernel_size=1, stride=1),
                BatchNorm2d(self.out_features),
                ReLU(),
                Conv2d(self.out_features, self.out_features, kernel_size=3, stride=self.stride, padding=1),
                BatchNorm2d(self.out_features),
                ReLU(),
                Conv2d(self.out_features, self.out_features * self.expansion, kernel_size=1, stride=1),
                BatchNorm2d(self.out_features * self.expansion))

        def __init_downsample(self):
            return self.__build_block() if self.__sizes_differ() else None

        def __sizes_differ(self):
            return self.stride != 1 or self.in_features != self.out_features * self.expansion

        def __build_block(self):
            return Sequential(
                Conv2d(self.in_features, self.out_features * self.expansion, kernel_size=1, stride=self.stride),
                BatchNorm2d(self.out_features * self.expansion))

        def forward(self, x):
            identity = x
            x = self.convolution(x)
            if self.downsample:
                identity = self.downsample(identity)
            return self.activation(x + identity)
