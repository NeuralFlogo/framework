import torch.optim

from discovery.hyperparameters.optimizer import Optimizer


class PytorchAdam(Optimizer):
    def __init__(self, model_params, lr):
        self.optimizer = torch.optim.Adam(model_params, lr=lr)
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
