from torch import nn

from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchUnidimensionalBatchNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(PytorchUnidimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum)


class PytorchBidimensionalBatchNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(PytorchBidimensionalBatchNormalizationLayer, self).__init__()
        self.layer = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)

