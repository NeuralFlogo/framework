class Kernel:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get(self):
        return self.x, self.y

# TODO simplificar channels in y out (secundario si se puede)


class ConvolutionalLayer:
    def __init__(self, in_channels: int, out_channels: int, kernel, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel
        self.stride = stride
        self.padding = padding
