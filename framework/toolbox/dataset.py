from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, batch_size: int): #TODO no se qué argumentos hacen falta como tal te lo dejo a tu elección
        self.seed = "random seed"
        self.batch_size = batch_size

    @abstractmethod
    def train(self) -> 'Dataset': #TODO no se si es un dataset lo que devuelve o una lista de batches, la cosa es que después de un train solo pueda obtener los batches
        pass

    @abstractmethod
    def test(self) -> 'Dataset':
        pass

    @abstractmethod
    def batches(self) -> 'Dataset':
        pass

    @abstractmethod
    def inputs(self): #TODO puede ir fuera en caso de que te crees una clase Batch
        pass

    @abstractmethod
    def targets(self):
        pass