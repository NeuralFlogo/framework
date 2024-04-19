from torch import Tensor
from torch.nn import Module

from framework.architecture.model import Model
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.toolbox.device import PytorchDevice


class PytorchModel(Module, Model):
    def __init__(self, architecture: PytorchArchitecture, device: PytorchDevice):
        Module.__init__(self)
        Model.__init__(self, architecture=architecture, device=device)
        self.architecture.to(device.get())

    def weights(self):
        return self.architecture.state_dict()

    def predict(self, x: Tensor) -> Tensor:
        self.architecture.eval()
        return self.architecture(x.unsqueeze(0).to(self.device.get())).squeeze(0).tolist()

    def forward(self, x: Tensor) -> Tensor:
        return self.architecture(x)