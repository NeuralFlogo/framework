import torch.optim

from framework.discovery.hyperparameters.optimizer import Optimizer


class PytorchSGD(Optimizer):
    def __init__(self, model_params, lr):
        self.optimizer = torch.optim.SGD(model_params, lr=lr)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
