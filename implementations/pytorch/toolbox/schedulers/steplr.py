from torch.optim.lr_scheduler import StepLR

from implementations.pytorch.toolbox.scheduler import PytorchScheduler


class PytorchStepLRScheduler(PytorchScheduler):
    def __init__(self, step_size: int, gamma: float, last_epoch=-1):
        super(PytorchStepLRScheduler, self).__init__()
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.scheduler = None

    def init(self, optimizer):
        self.scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=self.last_epoch)

    def move(self):
        self.scheduler.step()
