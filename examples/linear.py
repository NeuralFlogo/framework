from framework.toolbox.laboratory import Laboratory
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.blocks.linear import PytorchLinearBlock
from implementations.pytorch.architecture.layers.activations.relu import PytorchReluLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer
from implementations.pytorch.architecture.layers.regularizations.batch_normalization import \
    PytorchUnidimensionalBatchNormalizationLayer
from implementations.pytorch.architecture.layers.regularizations.dropout import PytorchDropoutLayer
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection
from implementations.pytorch.toolbox.datasets.builders.numeric_dataset_builder import PytorchNumericDatasetLoader
from implementations.pytorch.toolbox.experiment import PytorchExperiment
from implementations.pytorch.toolbox.losses.mse import MsePytorchLossFunction
from implementations.pytorch.toolbox.optimizers.sgd import SgdPytorchOptimizer
from implementations.pytorch.toolbox.checkpoint_saver import PytorchCheckpointSaver
from implementations.pytorch.toolbox.result_saver import PytorchResultSaver
from implementations.pytorch.toolbox.stopper import PytorchEarlyStopper
from implementations.pytorch.toolbox.strategies.regression import PytorchRegressionStrategy

PATH = "C:/Users/Joel/Desktop/winequality-red.csv"

dataset = PytorchNumericDatasetLoader(PATH, 5, 42).build(0.6, 0.2, 0.2)

architecture = (PytorchArchitecture("LinearArchitecture")
                        .attach(PytorchLinearSection(
                            [PytorchLinearBlock([
                                PytorchLinearLayer(12, 30),
                                PytorchUnidimensionalBatchNormalizationLayer(30, 0.5, 0.3),
                                PytorchReluLayer(),
                                PytorchDropoutLayer(0.5)]),
                                PytorchLinearBlock([
                                    PytorchLinearLayer(30, 10),
                                    PytorchUnidimensionalBatchNormalizationLayer(10, 0.5, 0.5),
                                    PytorchReluLayer(),
                                    PytorchDropoutLayer(0.4)]),
                                PytorchLinearBlock([
                                    PytorchLinearLayer(10, 1),
                                    PytorchReluLayer()])
                            ])))


experiment = PytorchExperiment("TestingExperiment", SgdPytorchOptimizer(architecture.parameters(), 0.001, 0.001),
                               MsePytorchLossFunction(), PytorchEarlyStopper(10, 0.01),
                               PytorchCheckpointSaver("C:/Users/Joel/Desktop/test/test.pt"))

loss = Laboratory("TestingLaboratory", 10, dataset, architecture, [experiment], PytorchRegressionStrategy(MsePytorchLossFunction()))\
    .explore(PytorchResultSaver("C:/Users/Joel/Desktop/test/result.tsv"))

print("The Lab loss is {}".format(loss))
