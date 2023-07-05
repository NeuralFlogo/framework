class ResidualLayer:

    def __init__(self, in_channels: int, out_channels: int, activation: str, stride=1, downsample=None, hidden_size=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.stride = stride
        self.downsample = downsample
        self.hidden_size = hidden_size
