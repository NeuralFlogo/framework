from torch import nn

from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchLayerNormalizationLayer(PytorchRegularizationLayer):
    def __init__(self, normalized_shape: int, eps: float):
        super(PytorchLayerNormalizationLayer, self).__init__()
        self.layer = nn.LayerNorm(normalized_shape=normalized_shape, eps=eps)
