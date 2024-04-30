import os

import torch

from framework.toolbox.saver import ModelSaver
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer

Delimiter = "/"
ModelFilename = "model.pt"
OptimizerFilename = "optimizer.pt"


class PytorchModelSaver(ModelSaver):
    def __init__(self, root: str):
        self.root = root
        self.__checkpoint_counter = 0

    def save(self, experiment: str, model: PytorchModel, optimizer: PytorchOptimizer):
        self.__checkpoint_counter += 1
        self.__init_experiment_folder(experiment)
        self.__save_weights(self.__path_of(experiment) + Delimiter + ModelFilename, model)
        if optimizer:
            self.__save_weights(self.__path_of(experiment) + Delimiter + OptimizerFilename, optimizer)

    def latest_checkpoint(self, experiment: str):
        return self.__path_of(experiment) + Delimiter + ModelFilename

    def __init_experiment_folder(self, experiment):
        if not self.__is_dir(self.__experiment_path(experiment)):
            self.__create_experiment_folder(experiment)

    def __save_weights(self, path: str, component: PytorchModel | PytorchOptimizer):
        if not self.__is_dir(os.path.dirname(path)):
            self.__mkdir(os.path.dirname(path))
        torch.save(component.weights(), path)

    def __path_of(self, experiment: str):
        return self.root + Delimiter + experiment + Delimiter + "checkpoint-" + str(self.__checkpoint_counter)

    def __is_dir(self, path):
        return os.path.exists(path)

    def __create_experiment_folder(self, experiment):
        self.__mkdir(self.__path_of(experiment))
        self.__create_manifest(self.__experiment_path(experiment), experiment)

    def __mkdir(self, path):
        os.makedirs(path)

    def __create_manifest(self, path, experiment):
        pass

    def __experiment_path(self, experiment):
        return self.root + Delimiter + experiment