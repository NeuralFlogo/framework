import json

import torch

from framework.toolbox.saver import CheckpointSaver
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer

MODEL_FILENAME = "model.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
JSON_EXTENSION = ".json"
CHECKPOINT_FILENAME = "model_checkpoint"


class PytorchCheckpointSaver(CheckpointSaver):
    def __init__(self, root: str):
        self.__checkpoint_counter = 0
        self.root = root
        self.__json = None

    def save(self, model: PytorchArchitecture, optimizer: PytorchOptimizer):
        self.__checkpoint_counter += 1
        self.__json = {}
        self.__save_model(model)
        if optimizer:
            self.__save_optimizer(optimizer)
        self.__save_json()

    def __save_model(self, model):
        torch.save(model.state_dict(), self.root + MODEL_FILENAME)
        self.__json["model"] = self.root + MODEL_FILENAME

    def __save_optimizer(self, optimizer):
        torch.save(optimizer.weights(), self.root + OPTIMIZER_FILENAME)
        self.__json["optimizer"] = self.root + OPTIMIZER_FILENAME

    def __save_json(self):
        with open(self.__json_model_name(), 'w') as file:
            json.dump(self.__json, file)

    def __json_model_name(self):
        return self.root + CHECKPOINT_FILENAME + str(self.__checkpoint_counter) + JSON_EXTENSION
