from typing import List

from framework.toolbox.laboratory import Laboratory
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.experiment import PytorchExperiment


class PytorchLaboratory(Laboratory):
    def __init__(self, epochs: int, dataset: PytorchDataset, architecture: PytorchArchitecture, experiments: List[PytorchExperiment]):
        super().__init__(epochs, dataset, architecture, experiments)

    def explore(self):
        pass