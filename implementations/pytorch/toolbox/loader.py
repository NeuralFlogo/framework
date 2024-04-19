import torch

from framework.toolbox.loader import ModelLoader
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.device import PytorchDevice


class PytorchModelLoader(ModelLoader):
    def load(self, path: str, architecture: PytorchArchitecture, device: PytorchDevice) -> PytorchModel:
        architecture.load_state_dict(torch.load(path))
        return PytorchModel(architecture=architecture, device=device)
