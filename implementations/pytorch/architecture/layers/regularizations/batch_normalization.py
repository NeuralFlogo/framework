from torch import nn, Tensor

from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchUnidimensionalBatchNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(PytorchUnidimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum, )

    def forward(self, x: Tensor) -> Tensor:
        if x.size(0) == 1:
            return x
        else:
            return self.layer(x)


class PytorchBidimensionalBatchNormalizationLayer(PytorchRegularizationLayer, nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(PytorchBidimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)


class PytorchMultidimensionalBatchNormalizationLayer(PytorchRegularizationLayer, nn.BatchNorm3d):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(PytorchMultidimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm3d(num_features=num_features, eps=eps, momentum=momentum)
