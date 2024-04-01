from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer
from implementations.pytorch.architecture.layers.activations.softmax import PytorchSoftmaxLayer
from implementations.pytorch.architecture.layers.convolutional import PytorchConvolutionalLayer
from implementations.pytorch.architecture.layers.flatten import PytorchFlattenLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer
from implementations.pytorch.architecture.layers.poolings.average import PytorchAvgPoolLayer
from implementations.pytorch.architecture.layers.poolings.max import PytorchMaxPoolLayer
from implementations.pytorch.architecture.layers.regularizations.batch_normalization import PytorchBidimensionalBatchNormalizationLayer
from implementations.pytorch.architecture.sections.convolutional import PytorchConvolutionalSection
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection

architecture = (Architecture("Convolutional")
                .attach(PytorchConvolutionalSection(
                            [Block([
                                 PytorchConvolutionalLayer(3, 33, 3, 2),
                                 PytorchBidimensionalBatchNormalizationLayer(33, 0.5, 0.3),
                                 PytorchReLULayer(),
                                 PytorchMaxPoolLayer(5, 4, 0)]),
                            Block([
                                 PytorchConvolutionalLayer(33, 16, 3, 3),
                                 PytorchBidimensionalBatchNormalizationLayer(16, 0.5, 0.3),
                                 PytorchReLULayer(),
                                 PytorchAvgPoolLayer(3, 2, 1)])])
                    )
                .attach(PytorchFlattenLayer())
                .attach(PytorchLinearSection([
                            Block([
                                PytorchLinearLayer(64, 1),
                                PytorchSoftmaxLayer(1)
                            ])])
                ))

print(architecture)
