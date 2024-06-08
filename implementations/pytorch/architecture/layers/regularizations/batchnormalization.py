from torch import nn, Tensor

from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class Pytorch1DimensionalBatchNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, num_features: int, eps: float, momentum: float):
        super(Pytorch1DimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum, )

    def forward(self, x: Tensor) -> Tensor:
        return x if x.size(0) == 1 else self.layer(x)


class Pytorch2DimensionalBatchNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(Pytorch2DimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)


class Pytorch3DimensionalBatchNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(Pytorch3DimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm3d(num_features=num_features, eps=eps, momentum=momentum)
