from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.blocks.linear import PytorchLinearBlock
from implementations.pytorch.architecture.layers.activations.relu import PytorchReluLayer
from implementations.pytorch.architecture.layers.convolutional import PytorchConvolutionalLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer
from implementations.pytorch.architecture.layers.regularizations.batch_normalization import \
    PytorchUnidimensionalBatchNormalizationLayer
from implementations.pytorch.architecture.layers.regularizations.dropout import PytorchDropoutLayer
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection

architecture = (PytorchArchitecture()
                .attach(PytorchLinearSection(
                    [PytorchLinearBlock([
                         PytorchLinearLayer(20, 30),
                         PytorchUnidimensionalBatchNormalizationLayer(30, 0.5, 0.3),
                         PytorchReluLayer(),
                         PytorchDropoutLayer(0.5)]),
                    PytorchLinearBlock([
                        PytorchLinearLayer(30, 40),
                        PytorchUnidimensionalBatchNormalizationLayer(40, 0.5, 0.5),
                        PytorchReluLayer(),
                        PytorchDropoutLayer(0.4)]),
                    PytorchLinearBlock([
                        PytorchLinearLayer(40, 50),
                        PytorchUnidimensionalBatchNormalizationLayer(50, 0.5, 0.5),
                        PytorchReluLayer(),
                        PytorchDropoutLayer(0.3)])
                    ])))
