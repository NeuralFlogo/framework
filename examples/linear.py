from framework.toolbox.laboratory import Laboratory
from framework.toolbox.logger import Logger
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer as ReLULayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer as LinearLayer
from implementations.pytorch.architecture.layers.regularizations.batchnormalization import Pytorch1DimensionalBatchNormalizationLayer as BatchNormalizationLayer
from implementations.pytorch.architecture.layers.regularizations.dropout import PytorchDropoutLayer as DropoutLayer
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection as LinearSection
from implementations.pytorch.toolbox.device import PytorchDevice as Device
from implementations.pytorch.toolbox.experiment import PytorchExperiment as Experiment
from implementations.pytorch.toolbox.data.generator import PytorchDatasetGenerator
from implementations.pytorch.toolbox.losses.mse import PytorchMSELossFunction as MSELossFunction
from implementations.pytorch.toolbox.optimizers.sgd import PytorchSGDOptimizer as SGDOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver as ModelSaver
from implementations.pytorch.toolbox.loader import PytorchModelLoader as ModelLoader
from implementations.pytorch.toolbox.strategies.regression import PytorchRegressionStrategy as RegressionStrategy

PATH = ""
DATASET_NAME = "winequality-red"

dataset = PytorchDatasetGenerator(DATASET_NAME, PATH, 10, 42).generate(0.7, 0.2, 0.1)


architecture = (Architecture("LinearArchitecture")
                    .attach(LinearSection([
                        Block([
                            LinearLayer(in_features=11, out_features=30, dimension=-1, bias=True),
                            BatchNormalizationLayer(num_features=30, eps=1.0E-5, momentum=0.3),
                            ReLULayer(),
                            DropoutLayer(probability=0.5)
                        ]),
                        Block([
                            LinearLayer(in_features=30, out_features=10, dimension=-1, bias=True),
                            BatchNormalizationLayer(num_features=10, eps=1.0E-5, momentum=0.5),
                            ReLULayer(),
                            DropoutLayer(probability=0.4)
                        ]),
                        Block([
                            LinearLayer(in_features=10, out_features=1, dimension=-1, bias=True),
                            ReLULayer()
                        ])
                    ])))

experiment = Experiment("r2d2",
                               SGDOptimizer(architecture.parameters(), learning_rate=0.001, momentum=0, dampening=0, weight_decay=0),
                               MSELossFunction(),
                               EarlyStopper(10, 0.01),
                               ModelSaver(""))

Laboratory("star-wars", 1, 10, dataset, architecture, [experiment], RegressionStrategy(MSELossFunction()), Logger(""), ModelLoader(), Device(1)).explore()
