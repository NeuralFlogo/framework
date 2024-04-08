from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.sections.recurrent import PytorchRecurrentSection as RecurrentSection
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection as LinearSection
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer as LinearLayer
from implementations.pytorch.architecture.layers.recurrents.lstm import PytorchLSTMLayer as LSTMLayer
from implementations.pytorch.architecture.layers.slicing import PytorchSlicingLayer as SlicingLayer
from implementations.pytorch.architecture.layers.flatten import PytorchFlattenLayer as FlattenLayer
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer as ReLULayer
from implementations.pytorch.architecture.layers.activations.softmax import PytorchSoftmaxLayer as SoftmaxLayer

architecture = (Architecture("RecurrentArchitecture")
                    .attach(RecurrentSection([
                        Block([
                            LSTMLayer(input_size=300, hidden_size=20, num_layer=4, bidirectional=True, dropout=0.5),
                            SlicingLayer(output=SlicingLayer.OutputType.EndSequence, start=4, end=7),
                            LinearLayer(in_features=120, out_features=1600, dimension=0, bias=True),
                            LinearLayer(in_features=40, out_features=20, dimension=1, bias=True)
                        ])
                    ]))
                    .attach(FlattenLayer(from_dim=1, to_dim=1))
                    .attach(LinearSection([
                        Block([
                            LinearLayer(in_features=20, out_features=150, dimension=-1, bias=True),
                            ReLULayer()
                        ]),
                        Block([
                            LinearLayer(in_features=150, out_features=10, dimension=-1, bias=True),
                            ReLULayer(),
                            SoftmaxLayer(n_dimensions=1)
                        ])
                    ])))
