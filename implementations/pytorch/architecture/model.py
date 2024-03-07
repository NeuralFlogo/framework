from torch.nn import Module

from framework.architecture.model import Model
from implementations.pytorch.architecture.architecture import PytorchArchitecture


class PytorchModel(Module, Model):
    def __init__(self, architecture: PytorchArchitecture):
        Module.__init__(self)
        Model.__init__(self, architecture)

    def weights(self):
        return self.architecture.state_dict()

    def predict(self):
        pass

    def eval(self):
        return self.architecture.eval()