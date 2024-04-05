from torch.nn import HingeEmbeddingLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchHingeEmbeddingLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchHingeEmbeddingLossFunction, self).__init__()
        self.loss = HingeEmbeddingLoss()
        