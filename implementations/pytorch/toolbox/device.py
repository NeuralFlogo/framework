import torch

from framework.toolbox.device import Device


class PytorchDevice(Device):
    def __init__(self, device: int):
        super().__init__(device)
        self.devices = {-1: "default", 0: "cpu", 1: "cuda", 2: "mps"}

    def get(self):
        if self.device == -1:
            if torch.cuda.is_available(): return self.devices.get(1)
            if torch.backends.mps.is_available(): return self.devices.get(2)
            return self.devices.get(0)
        return self.devices.get(self.device)
