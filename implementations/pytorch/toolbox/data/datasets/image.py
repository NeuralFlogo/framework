import torch
from PIL import Image
from torchvision import transforms

from implementations.pytorch.toolbox.data.dataset import PytorchDataset


class PytorchImageDataset(PytorchDataset):
    def __init__(self, batch_size: int, images):
        super().__init__(batch_size)
        self.__images = images
        self.__tensor_transformer = transforms.ToTensor()

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, idx):
        image = self.__images[idx]
        image_path, label = image[0], image[1]
        image = Image.open(image_path)
        return self.__tensor_transformer(image), torch.tensor(label, dtype=torch.long)
