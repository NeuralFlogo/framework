from layers.residual import ResidualLayer


class PytorchResidual(ResidualLayer):
    def __init__(self, in_channels: int, out_channels: int, activation: str):
        super().__init__(in_channels, out_channels, activation)

