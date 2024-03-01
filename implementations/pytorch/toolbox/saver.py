import torch

from framework.toolbox.saver import Saver
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchSaver(Saver):
    def __init__(self, root: str):
        self.root = root

    def save(self, model: PytorchArchitecture, optimizer: PytorchOptimizer):
        checkpoint = {'model_state_dict': model.state_dict()}
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.params()
        torch.save(checkpoint, self.root)
