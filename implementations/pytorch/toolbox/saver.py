import os

import torch

from framework.architecture.model import Model
from framework.toolbox.experiment import Experiment
from framework.toolbox.saver import ModelSaver

PATH_DELIMITER = "/"
MODEL_FILENAME = "model.pt"
OPTIMIZER_FILENAME = "optimizer.pt"


class PytorchModelSaver(ModelSaver):
    def __init__(self, root: str):
        self.__checkpoint_counter = 0
        self.root = root
        self.__json = None

    def save(self, model: Model, experiment: Experiment):
        self.__checkpoint_counter += 1
        path = self.__create_path(experiment)
        self.__check_experiment_manifest(path, experiment)
        self.__save_weigth(path + PATH_DELIMITER + MODEL_FILENAME, model)
        if experiment.optimizer:
            self.__save_weigth(path + PATH_DELIMITER + OPTIMIZER_FILENAME, experiment.optimizer)

    def __check_experiment_manifest(self, path, experiment):
        if not self.__is_folder(os.path.dirname(path)):
            self.__create_experiment_folder(os.path.dirname(path), experiment)

    def __save_weigth(self, path, model):
        torch.save(model.weights(), path)

    def __create_experiment_folder(self, path, experiment):
        self.__create_folder(path)
        self.__create_experiment_manifest(path, experiment)

    def __create_experiment_manifest(self, path, experiment):
        pass

    def __is_folder(self, path):
        return os.path.exists(path)

    def __create_folder(self, path):
        os.makedirs(path)

    def __create_path(self, experiment):
        return self.root + PATH_DELIMITER + experiment.type_measurement + PATH_DELIMITER + self.__checkpoint_counter
