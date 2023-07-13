from architecture import Architecture
from blocks.linear import LinearBlock
from input import Input
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftmax
from pytorch.layers.linear import PytorchLinear
from sections.linear import LinearSection


inputs = Input(shape=(6, 6, 6))
outputs = LinearSection([LinearBlock(PytorchLinear(1600, 120), PytorchRelu()), LinearBlock(PytorchLinear(1600, 120), PytorchSoftmax())])

architecture = Architecture(inputs=inputs, outputs=outputs, name="Linear")
