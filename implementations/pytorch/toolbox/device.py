from framework.toolbox.device import Device


class PytorchDevice(Device):
    def __init__(self, device: int):
        super().__init__(device)
        self.devices = {0: "cpu", 1: "cuda", 2: "mps"}

    def get(self):
        return self.devices.get(self.device)
