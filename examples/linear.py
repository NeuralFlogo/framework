from generics.architecture import Architecture
from blocks.linear import LinearBlock
from generics.data import Data
from pytorch.layers.linear import PytorchLinear
from sections.linear import LinearSection


data = Data(shape=(3, 3, 3))

data = LinearSection([
    LinearBlock(PytorchLinear(1600, 120))])(data)

data = LinearSection([
    LinearBlock(PytorchLinear(1600, 120))])(data)

architecture = Architecture(data=data, name="Linear").build()

print(architecture.architecture)
