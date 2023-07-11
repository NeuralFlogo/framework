from architecture import Architecture
from blocks.convolutional import ConvolutionalBlock
from blocks.linear import LinearBlock
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftmax
from pytorch.layers.convolutional import PytorchConvolutional
from pytorch.layers.flatten import PytorchFlatten
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.poolings.max import PytorchMaxPooling
from pytorch.network import PytorchNetwork
from sections.convolutional import ConvolutionalSection
from sections.linear import LinearSection

layers = [
    ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(1, 6, 5), PytorchRelu())], PytorchMaxPooling()),
    ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(7, 8, 5), PytorchRelu())], PytorchMaxPooling()),
    PytorchFlatten(1, 3),
    LinearSection([LinearBlock(PytorchLinear(1600, 120), PytorchRelu()), LinearBlock(PytorchLinear(1600, 120), PytorchSoftmax())])
]

model = PytorchNetwork(Architecture(layers))

print(model.parameters())
