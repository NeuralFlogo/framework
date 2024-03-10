from framework.toolbox.laboratory import Laboratory
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.blocks.linear import PytorchLinearBlock
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer
from implementations.pytorch.architecture.layers.regularizations.batch_normalization import \
    PytorchUnidimensionalBatchNormalizationLayer
from implementations.pytorch.architecture.layers.regularizations.dropout import PytorchDropoutLayer
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection
from implementations.pytorch.toolbox.loaders.numeric import PytorchNumericDatasetLoader
from implementations.pytorch.toolbox.experiment import PytorchExperiment
from implementations.pytorch.toolbox.losses.mse import PytorchMSELossFunction
from implementations.pytorch.toolbox.optimizers.sgd import PytorchSGDOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver
from implementations.pytorch.toolbox.logger import PytorchLogger
from implementations.pytorch.toolbox.strategies.regression import PytorchRegressionStrategy

PATH = "C:/Users/juanc/Downloads/winequality-red.csv"

dataset = PytorchNumericDatasetLoader(PATH, 5, 42).load(0.6, 0.2, 0.2)

architecture = (PytorchArchitecture("LinearArchitecture")
                    .attach(PytorchLinearSection(
                        [PytorchLinearBlock([
                            PytorchLinearLayer(12, 30),
                            PytorchUnidimensionalBatchNormalizationLayer(30, 0.5, 0.3),
                            PytorchReLULayer(),
                            PytorchDropoutLayer(0.5)]),
                            PytorchLinearBlock([
                                PytorchLinearLayer(30, 10),
                                PytorchUnidimensionalBatchNormalizationLayer(10, 0.5, 0.5),
                                PytorchReLULayer(),
                                PytorchDropoutLayer(0.4)]),
                            PytorchLinearBlock([
                                PytorchLinearLayer(10, 1),
                                PytorchReLULayer()])
                        ])))

experiment = PytorchExperiment("TestingExperiment",
                               PytorchSGDOptimizer(architecture.parameters(), 0.001, 0.001),
                               PytorchMSELossFunction(),
                               EarlyStopper(10, 0.01),
                               PytorchModelSaver("C:/Users/juanc/Downloads/folder/test.pt"))

loss = (Laboratory(name="TestingLaboratory",
                   epochs=10,
                   dataset=dataset,
                   architecture=architecture,
                   experiments=[experiment],
                   strategy=PytorchRegressionStrategy(PytorchMSELossFunction()),
                   logger=PytorchLogger("C:/Users/juanc/Downloads/folder/result.tsv"))
        .explore())

print("The Lab loss is {}".format(loss))
