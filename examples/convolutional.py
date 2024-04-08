from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.sections.convolutional import PytorchConvolutionalSection as ConvolutionalSection
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection as LinearSection
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.layers.convolutional import Pytorch2DimensionalConvolutionalLayer as ConvolutionalLayer
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer as ReLULayer
from implementations.pytorch.architecture.layers.poolings.maxpool import Pytorch2DimensionalMaxPoolLayer as MaxPoolLayer
from implementations.pytorch.architecture.layers.flatten import PytorchFlattenLayer as FlattenLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer as LinearLayer
from implementations.pytorch.architecture.layers.activations.softmax import PytorchSoftmaxLayer as SoftmaxLayer

architecture = (Architecture("ConvolutionalArchitecture")
                    .attach(ConvolutionalSection([
                        Block([
                            ConvolutionalLayer(in_channels=3, out_channels=40, kernel=(26, 26), stride=(1, 1), padding=(0, 0)),
                            ReLULayer(),
                            MaxPoolLayer(kernel=(2, 2), stride=(2, 2), padding=(13, 13))
                        ]),
                        Block([
                            ConvolutionalLayer(in_channels=40, out_channels=80, kernel=(26, 26), stride=(1, 1), padding=(0, 0)),
                            ReLULayer(),
                            MaxPoolLayer(kernel=(3, 3), stride=(3, 3), padding=(2, 2))
                        ])
                    ]))
                    .attach(FlattenLayer(from_dim=3, to_dim=1))
                    .attach(LinearSection([
                        Block([
                            LinearLayer(in_features=8000, out_features=150, dimension=-1, bias=True),
                            ReLULayer()
                        ]),
                        Block([
                            LinearLayer(in_features=150, out_features=10, dimension=-1, bias=True),
                            ReLULayer(),
                            SoftmaxLayer(n_dimensions=1)
                        ])
                    ])))


print(architecture)