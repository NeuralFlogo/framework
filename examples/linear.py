from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection as LinearSection
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer as LinearLayer
from implementations.pytorch.architecture.layers.regularizations.batchnormalization import Pytorch1DimensionalBatchNormalizationLayer as BatchNormalizationLayer
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer as ReLULayer
from implementations.pytorch.architecture.layers.regularizations.dropout import PytorchDropoutLayer as DropoutLayer

architecture = (Architecture("LinearArchitecture")
                    .attach(LinearSection([
                        Block([
                            LinearLayer(in_features=12, out_features=30, dimension=-1, bias=True),
                            BatchNormalizationLayer(num_features=30, eps=1.0E-5, momentum=0.1),
                            ReLULayer(),
                            DropoutLayer(probability=0.8)
                        ]),
                        Block([
                            LinearLayer(in_features=30, out_features=10, dimension=-1, bias=True),
                            BatchNormalizationLayer(num_features=10, eps=1.0E-5, momentum=0.1),
                            ReLULayer(),
                            DropoutLayer(probability=0.8)
                        ]),
                        Block([
                            LinearLayer(in_features=10, out_features=1, dimension=-1, bias=True),
                            ReLULayer()
                        ])
                    ])))

print(architecture)
