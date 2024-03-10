import json
import os

import torch

from framework.toolbox.saver import CheckpointSaver
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer

MODEL_FILENAME = "model.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
JSON_FILENAME = "checkpoint.json"
CHECKPOINT_FOLDER = "model_checkpoint"


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
        torch.save(model.state_dict(), self.__folder() + MODEL_FILENAME)
        self.__json["model"] = self.__folder() + MODEL_FILENAME

    def __save_optimizer(self, optimizer):
        torch.save(optimizer.weights(), self.__folder() + OPTIMIZER_FILENAME)
        self.__json["optimizer"] = self.__folder() + OPTIMIZER_FILENAME

    def __save_json(self):
        with open(self.__folder() + JSON_FILENAME, 'w') as file:
            json.dump(self.__json, file)

    def __folder(self):
        folder = self.root + CHECKPOINT_FOLDER + str(self.__checkpoint_counter) + "/"
        if not os.path.exists(folder): os.makedirs(folder)
        return folder
