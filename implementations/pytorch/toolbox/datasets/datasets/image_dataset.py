from typing import List

import torch
from torch.utils.data import DataLoader

from framework.toolbox.batch import Batch
from framework.toolbox.dataset import Dataset
from implementations.pytorch.toolbox.datasets.batch import PytorchBatch
from PIL import Image
from torchvision import transforms


class PytorchImageDataset(Dataset, torch.utils.data.Dataset):

    def __init__(self, batch_size, images):
        super().__init__(batch_size)
        self.__images = images
        self.__tensor_transformer = transforms.ToTensor()

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, idx):
        image = self.__images[idx]
        image_path, label = image[0], image[1]
        image = Image.open(image_path)
        return self.__tensor_transformer(image), torch.tensor(label)

    def batches(self) -> List[Batch]:
        batches = []
        for inputs, targets in DataLoader(self, batch_size=self.batch_size, shuffle=True):
            batches.append(PytorchBatch(inputs, targets))
        return batches
