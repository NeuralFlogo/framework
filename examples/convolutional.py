from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.blocks.convolutional import PytorchConvolutionalBlock
from implementations.pytorch.architecture.blocks.linear import PytorchLinearBlock
from implementations.pytorch.architecture.layers.activations.relu import PytorchReluLayer
from implementations.pytorch.architecture.layers.activations.softmax import PytorchSoftmaxLayer
from implementations.pytorch.architecture.layers.convolutional import PytorchConvolutionalLayer
from implementations.pytorch.architecture.layers.flatten import PytorchFlattenLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer
from implementations.pytorch.architecture.layers.poolings.average import PytorchAveragePoolingLayer
from implementations.pytorch.architecture.layers.poolings.max import PytorchMaxPoolingLayer
from implementations.pytorch.architecture.layers.regularizations.batch_normalization import \
    PytorchBidimensionalBatchNormalizationLayer, PytorchUnidimensionalBatchNormalizationLayer
from implementations.pytorch.architecture.sections.convolutional import PytorchConvolutionalSection
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection
from implementations.pytorch.toolbox.loaders.image import PytorchImageDatasetLoader

PATH = "C:/Users/Joel/Desktop/PetImages"

dataset = PytorchImageDatasetLoader(PATH, 5, 42).load(0.6, 0.2, 0.2)

for batch in dataset.train().batches():
    print(batch.inputs())

# architecture = (PytorchArchitecture()
#                 .attach(PytorchConvolutionalSection(
#                             [PytorchConvolutionalBlock([
#                                  PytorchConvolutionalLayer(16, 33, 3, 2),
#                                  PytorchBidimensionalBatchNormalizationLayer(33, 0.5, 0.3),
#                                  PytorchReluLayer(),
#                                  PytorchMaxPoolingLayer(5, 4, 0)]),
#                             PytorchConvolutionalBlock([
#                                  PytorchConvolutionalLayer(33, 16, 3, 3),
#                                  PytorchBidimensionalBatchNormalizationLayer(16, 0.5, 0.3),
#                                  PytorchReluLayer(),
#                                  PytorchAveragePoolingLayer(3, 2, 1)])])
#                     )
#                 .attach(PytorchFlattenLayer())
#                 .attach(PytorchLinearSection([
#                             PytorchLinearBlock([
#                                 PytorchLinearLayer(32, 20),
#                                 PytorchUnidimensionalBatchNormalizationLayer(20, 0.5, 0.3),
#                                 PytorchSoftmaxLayer(1)
#                             ])])
#                 ))

