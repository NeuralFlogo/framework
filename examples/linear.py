from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection as LinearSection
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer as LinearLayer
from implementations.pytorch.architecture.layers.regularizations.batch_norm import PytorchUnidimensionalBatchNormalizationLayer as UnidimensionalBatchNormalizationLayer
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer as ReLULayer
from implementations.pytorch.architecture.layers.regularizations.dropout import PytorchDropoutLayer as DropoutLayer

architecture = (Architecture("Linear")
                    .attach(LinearSection(
                        [Block([
                            LinearLayer(12, 30),
                            UnidimensionalBatchNormalizationLayer(30, 0.5, 0.3),
                            ReLULayer(),
                            DropoutLayer(0.5)]),
                            Block([
                                LinearLayer(30, 10),
                                UnidimensionalBatchNormalizationLayer(10, 0.5, 0.5),
                                ReLULayer(),
                                DropoutLayer(0.4)]),
                            Block([
                                LinearLayer(10, 1),
                                ReLULayer()])
                        ])))

print(architecture)
