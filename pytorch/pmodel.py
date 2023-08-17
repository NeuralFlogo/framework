from torch.nn import Module, Sequential

from model import Model


class PytorchModel(Model, Module):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.module_list = None

    def build(self, layers):
        self.module_list = Sequential(*layers)
        return self

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x
