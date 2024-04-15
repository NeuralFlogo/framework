from sklearn.model_selection import train_test_split

from framework.toolbox.dataset import Dataset
from framework.toolbox.generator import DatasetGenerator
from framework.toolbox.loader import DatasetLoader
from implementations.pytorch.toolbox.loaders.image import PytorchImageDatasetLoader
from implementations.pytorch.toolbox.loaders.numeric import PytorchNumericDatasetLoader

META_DATASET_FILE_NAME = "meta-dataset.tsv"
META_DATASET_FILE_DELIMITER = "\t"
LOADER_DICT = {
    "numeric": PytorchNumericDatasetLoader,
    "image": PytorchImageDatasetLoader
}


class PytorchDatasetGenerator(DatasetGenerator):

    def __init__(self, name: str, path: str, batch_size: int, seed: int):
        super().__init__(name, path, batch_size, seed)
        self.__metadata = self.__read_metadata()
        self.__loader = self.__get_loader()
        self.__dataset = self.__loader.load()

    def generate(self, train_proportion: float, validation_proportion: float,
                 test_proportion: float) -> 'DatasetGenerator':
        train_data, test_data = train_test_split(self.__dataset, test_size=test_proportion, random_state=self.seed)
        train_data, val_data = train_test_split(train_data, test_size=validation_proportion, random_state=self.seed)
        self.datasets.append(self.__loader.create_dataset(self.batch_size, train_data))
        self.datasets.append(self.__loader.create_dataset(self.batch_size, val_data))
        self.datasets.append(self.__loader.create_dataset(self.batch_size, test_data))
        return self

    def train(self) -> 'Dataset':
        return self.datasets[0]

    def validation(self) -> 'Dataset':
        return self.datasets[1]

    def test(self) -> 'Dataset':
        return self.datasets[2]

    def __read_metadata(self):
        dataset_metadata = {}
        with open(self.path + META_DATASET_FILE_NAME, "r") as file:
            for line in file:
                line_array = line.split(META_DATASET_FILE_DELIMITER)
                dataset_metadata[line_array[0].lower()] = line_array[1].lower().strip()
        return dataset_metadata

    def __get_loader(self) -> DatasetLoader:
        return LOADER_DICT[self.__metadata["dataset"]](self.path, self.name, self.seed, self.__metadata)
