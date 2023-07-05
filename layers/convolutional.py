class ConvolutionalLayer:

    def __init__(self, in_channels: int, out_channels: int, kernel, stride, padding):
        self.kernel_size = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
