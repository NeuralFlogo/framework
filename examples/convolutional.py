from architecture import Architecture
from blocks.convolutional import ConvolutionalBlock
from blocks.linear import LinearBlock
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftmax
from pytorch.layers.convolutional import PytorchConvolutional
from pytorch.layers.flatten import PytorchFlatten
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.normalizations.batch_normalization import PytorchBatchNormalization
from pytorch.layers.normalizations.dropout import PytorchDropout
from pytorch.layers.poolings.max import PytorchMaxPooling
from pytorch.pmodel import PytorchModel
from sections.convolutional import ConvolutionalSection
from sections.linear import LinearSection

model = Architecture(PytorchModel(), "convolutional") \
    .attach(ConvolutionalSection([
     ConvolutionalBlock(PytorchConvolutional(1, 6, 5), PytorchRelu())], PytorchMaxPooling())) \
    .attach(ConvolutionalSection([
     ConvolutionalBlock(PytorchConvolutional(7, 8, 5), PytorchRelu())], PytorchMaxPooling())) \
    .attach(PytorchFlatten(1, 3)) \
    .attach(LinearSection([
     LinearBlock(PytorchLinear(110, 60), [PytorchBatchNormalization(110)], PytorchRelu(), [PytorchDropout(0.5)]),
     LinearBlock(PytorchLinear(60, 30), [PytorchBatchNormalization(60)], PytorchRelu(), [PytorchDropout(0.5)]),
     LinearBlock(PytorchLinear(30, 2), [PytorchBatchNormalization(30)], PytorchSoftmax(2))])) \
    .build()
