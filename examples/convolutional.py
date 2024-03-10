from framework.toolbox.laboratory import Laboratory
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.blocks.convolutional import PytorchConvolutionalBlock
from implementations.pytorch.architecture.blocks.linear import PytorchLinearBlock
from implementations.pytorch.architecture.layers.activations.relu import PytorchReLULayer
from implementations.pytorch.architecture.layers.activations.softmax import PytorchSoftmaxLayer
from implementations.pytorch.architecture.layers.convolutional import PytorchConvolutionalLayer
from implementations.pytorch.architecture.layers.flatten import PytorchFlattenLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer
from implementations.pytorch.architecture.layers.poolings.average import PytorchAveragePoolingLayer
from implementations.pytorch.architecture.layers.poolings.max import PytorchMaxPoolingLayer
from implementations.pytorch.architecture.layers.regularizations.batch_normalization import \
    PytorchBidimensionalBatchNormalizationLayer
from implementations.pytorch.architecture.sections.convolutional import PytorchConvolutionalSection
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection
from implementations.pytorch.toolbox.experiment import PytorchExperiment
from implementations.pytorch.toolbox.loaders.image import PytorchImageDatasetLoader
from implementations.pytorch.toolbox.logger import PytorchLogger
from implementations.pytorch.toolbox.losses.mse import PytorchMSELossFunction
from implementations.pytorch.toolbox.optimizers.sgd import PytorchSGDOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver
from implementations.pytorch.toolbox.strategies.classification import PytorchClassificationStrategy

PATH = "C:/Users/juanc/Downloads/PetImages/PetImages/"

dataset = PytorchImageDatasetLoader(PATH, 53, 42).load(0.6, 0.2, 0.2)

architecture = (PytorchArchitecture("Convolutional")
                .attach(PytorchConvolutionalSection(
                            [PytorchConvolutionalBlock([
                                 PytorchConvolutionalLayer(3, 33, 3, 2),
                                 PytorchBidimensionalBatchNormalizationLayer(33, 0.5, 0.3),
                                 PytorchReLULayer(),
                                 PytorchMaxPoolingLayer(5, 4, 0)]),
                            PytorchConvolutionalBlock([
                                 PytorchConvolutionalLayer(33, 16, 3, 3),
                                 PytorchBidimensionalBatchNormalizationLayer(16, 0.5, 0.3),
                                 PytorchReLULayer(),
                                 PytorchAveragePoolingLayer(3, 2, 1)])])
                    )
                .attach(PytorchFlattenLayer())
                .attach(PytorchLinearSection([
                            PytorchLinearBlock([
                                PytorchLinearLayer(64, 1),
                                PytorchSoftmaxLayer(1)
                            ])])
                ))

experiment = PytorchExperiment("ConvolutionalExperiment",
                               PytorchSGDOptimizer(architecture.parameters(), 0.001, 0.001),
                               PytorchMSELossFunction(),
                               EarlyStopper(5, 0.1),
                               PytorchModelSaver("C:/Users/juanc/Downloads/folder/"))

loss = (Laboratory(name="ConvolutionalLaboratory",
                   epochs=10,
                   dataset=dataset,
                   architecture=architecture,
                   experiments=[experiment],
                   strategy=PytorchClassificationStrategy(),
                   logger=PytorchLogger("C:/Users/juanc/Downloads/folder/result.tsv"))
        .explore())

print("Predicted", loss[0])
print("Actual", loss[1])
print("The Lab loss is {}".format(loss))
