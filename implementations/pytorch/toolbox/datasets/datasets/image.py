from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from implementations.pytorch.toolbox.dataset import PytorchDataset


class PytorchImageDataset(PytorchDataset, Dataset):
    def __init__(self, batch_size, images):
        super().__init__(batch_size)
        self.__images = images
        self.__tensor_transformer = transforms.ToTensor()
        self.__num_classes = 0

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, idx):
        image = self.__images[idx]
        label = [0] * self.__num_classes
        image_path, label[image[1]] = image[0], 1
        image = Image.open(image_path)
        return self.__tensor_transformer(image), torch.tensor(label, dtype=torch.float32)

    def batches(self) -> List[PytorchDataset.PytorchBatch]:
        batches = []
        for inputs, targets in DataLoader(self, batch_size=self.batch_size, shuffle=True):
            batches.append(PytorchDataset.PytorchBatch(inputs, targets))
        return batches

    def set_num_classes(self, num_classes):
        self.__num_classes = num_classes
        return self
