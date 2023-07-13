from architecture import Architecture
from blocks.convolutional import ConvolutionalBlock
from blocks.linear import LinearBlock
from input import Input
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftmax
from pytorch.layers.convolutional import PytorchConvolutional
from pytorch.layers.flatten import PytorchFlatten
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.poolings.max import PytorchMaxPooling
from sections.convolutional import ConvolutionalSection
from sections.linear import LinearSection


inputs = Input(shape=(6, 6, 6))
x = ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(1, 6, 5), PytorchRelu())], PytorchMaxPooling())(inputs)
x = ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(7, 8, 5), PytorchRelu())], PytorchMaxPooling())(x)
x = PytorchFlatten(1, 3)(x)
outputs = LinearSection([LinearBlock(PytorchLinear(1600, 120), PytorchRelu()), LinearBlock(PytorchLinear(1600, 120), PytorchSoftmax())])(x)

architecture = Architecture(inputs=inputs, outputs=outputs, name="Convolutional")