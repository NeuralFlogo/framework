from architecture import Architecture
from blocks.convolutional import ConvolutionalBlock
from blocks.linear import LinearBlock
from blocks.residual import ResidualBlock
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftmax
from pytorch.layers.convolutional import PytorchConvolutional
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.normalizations.batch_normalization import PytorchBatchNormalization
from pytorch.layers.normalizations.dropout import PytorchDropout
from pytorch.layers.poolings.average import PytorchAveragePooling
from pytorch.layers.poolings.max import PytorchMaxPooling
from pytorch.layers.residual import PytorchResidual
from pytorch.pmodel import PytorchModel
from sections.convolutional import ConvolutionalSection
from sections.linear import LinearSection
from sections.residual import ResidualSection

model = Architecture(PytorchModel(), "resnet")\
    .attach(ConvolutionalSection([
                ConvolutionalBlock(PytorchConvolutional(3, 64, 7, 2, 3), PytorchBatchNormalization(64), PytorchRelu())],
                PytorchMaxPooling(3, 2, 1)))\
    .attach(ResidualSection(
                ResidualBlock([PytorchResidual(64, 1, 3), PytorchResidual(128, 2, 4), PytorchResidual(256, 2, 6), PytorchResidual(512, 2, 3)]),
                PytorchAveragePooling(7, 1, 0)))\
    .attach(LinearSection([
                LinearBlock(PytorchLinear(512, 1000), [PytorchBatchNormalization(512)], PytorchRelu(), [PytorchDropout(0.5)]),
                LinearBlock(PytorchLinear(1000, 10), [PytorchBatchNormalization(512)], PytorchSoftmax(10))])).build()

print(model)
