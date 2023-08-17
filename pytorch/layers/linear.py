from torch import nn

from layers.linear import LinearLayer


class PytorchLinear(LinearLayer):
    def __init__(self, in_features, out_features):
        self.layer = nn.Linear(in_features=in_features, out_features=out_features)

    def get(self):
        return self.layer
