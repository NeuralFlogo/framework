from architecture import Architecture
from blocks.linear import LinearBlock
from input import Input
from pytorch.layers.linear import PytorchLinear
from sections.linear import LinearSection


data = Input(shape=(3, 3, 3))

data = LinearSection([
    LinearBlock(PytorchLinear(1600, 120))])(data)

data = LinearSection([
    LinearBlock(PytorchLinear(1600, 120))])(data)

architecture = Architecture(input=data, name="Linear").build()

print(architecture.architecture)
