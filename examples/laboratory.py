import architecture
from framework.toolbox.laboratory import Laboratory
from framework.toolbox.logger import Logger
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.toolbox.experiment import PytorchExperiment
from implementations.pytorch.toolbox.optimizers.sgd import PytorchSGDOptimizer
from implementations.pytorch.toolbox.losses.mse import PytorchMSELossFunction
from implementations.pytorch.toolbox.loaders.numeric import PytorchNumericDatasetLoader
from implementations.pytorch.toolbox.saver import PytorchModelSaver
from implementations.pytorch.toolbox.strategies.regression import PytorchRegressionStrategy

PATH = ""

dataset = PytorchNumericDatasetLoader(PATH, 10, 42).load(0.7, 0.1, 0.2)

experiments = [PytorchExperiment("experiment test",
                                 PytorchSGDOptimizer(architecture.architecture.parameters(), 0.001, 0.01),
                                 PytorchMSELossFunction(),
                                 EarlyStopper(10, 0.01),
                                 PytorchModelSaver("C:/Users/juanc/Downloads/folder/test.pt")),
               PytorchExperiment("experiment test",
                                 PytorchSGDOptimizer(architecture.architecture.parameters(), 0.001, 0.01),
                                 PytorchMSELossFunction(),
                                 EarlyStopper(10, 0.01),
                                 PytorchModelSaver("C:/Users/juanc/Downloads/folder/test.pt"))]

loss = (Laboratory("LaboratoryName",
                   epochs=10,
                   dataset=dataset,
                   architecture=architecture.architecture,
                   experiments=experiments,
                   strategy=PytorchRegressionStrategy(),
                   logger=Logger("C:/Users/juanc/Downloads/folder/result.tsv"))
        .explore())

print("The Lab loss is {}".format(loss))
