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
        self.__checkpoint_counter = 1
        self.root = root

    def save(self, experiment: str, model: PytorchModel, optimizer: PytorchOptimizer):
        self.__init_manifest(experiment)
        self.__save_weigths(self.__path_of(experiment) + Delimiter + ModelFilename, model)
        if optimizer:
            self.__save_weigths(self.__path_of(experiment) + Delimiter + OptimizerFilename, optimizer)
        self.__checkpoint_counter += 1

    def __init_manifest(self, experiment):
        if not self.__is_dir(self.root + Delimiter + experiment):
            self.__mkdir(self.root + Delimiter + experiment)

    def __save_weigths(self, path: str, component: PytorchModel | PytorchOptimizer):
        torch.save(component.weights(), path)

    def __path_of(self, experiment: str):
        return self.root + Delimiter + experiment + Delimiter + str(self.__checkpoint_counter)

    def __is_dir(self, path):
        return os.path.exists(path)

    def __create_experiment_folder(self, experiment):
        self.__mkdir(self.__path_of(experiment))
        self.__create_manifest(self.__path_of(experiment), experiment)

    def __mkdir(self, path):
        os.makedirs(path)

    def __create_manifest(self, path, experiment):
        pass
