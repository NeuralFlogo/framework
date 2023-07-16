from blocks.convolutional import ConvolutionalBlock
from blocks.linear import LinearBlock
from architecture import Architecture
from input import Input
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftMax
from pytorch.layers.convolutional import PytorchConvolutional
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.poolings.max import PytorchMaxPooling
from sections.convolutional import ConvolutionalSection
from sections.linear import LinearSection

data = Input(shape=(6, 6, 6))
data = ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(1, 6, 5), PytorchRelu())], PytorchMaxPooling())(data)
data = ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(7, 8, 5), PytorchRelu())], PytorchMaxPooling())(data)
# data = PytorchFlatten(1, 3)(x)
outputs = LinearSection([LinearBlock(PytorchLinear(1600, 120), PytorchRelu()), LinearBlock(PytorchLinear(1600, 120), PytorchSoftMax(2))])(data)

architecture = Architecture(input=data, name="Linear").build()

print(architecture.architecture)
