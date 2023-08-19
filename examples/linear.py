from architecture import Architecture
from blocks.linear import LinearBlock
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftmax
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.normalizations.batch_normalization import PytorchBatchNormalization
from pytorch.layers.normalizations.dropout import PytorchDropout
from pytorch.pmodel import PytorchModel
from sections.linear import LinearSection

model = Architecture(network=PytorchModel(), name="Linear") \
    .attach(LinearSection([
    LinearBlock(PytorchLinear(112, 50), [PytorchBatchNormalization(512)], PytorchRelu(), [PytorchDropout(0.5)]),
    LinearBlock(PytorchLinear(50, 20), [PytorchBatchNormalization(50)], PytorchRelu(), [PytorchDropout(0.5)]),
    LinearBlock(PytorchLinear(20, 10), [PytorchBatchNormalization(512)], PytorchSoftmax(10))])) \
    .build()
print(model)