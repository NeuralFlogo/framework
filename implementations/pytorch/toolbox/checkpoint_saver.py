import torch

from framework.toolbox.checkpoint_saver import CheckpointSaver
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchCheckpointSaver(CheckpointSaver):
    def __init__(self, root: str):
        self.root = root

    def save(self, model: PytorchArchitecture, optimizer: PytorchOptimizer):
        checkpoint = {'model_state_dict': model.state_dict()}
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.weights()
        torch.save(checkpoint, self.root)
