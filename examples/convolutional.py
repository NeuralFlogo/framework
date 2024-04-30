from framework.toolbox.laboratory import Laboratory
from framework.toolbox.logger import Logger
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.layers.activations.relu import PytorchReluLayer as ReLULayer
from implementations.pytorch.architecture.layers.activations.softmax import PytorchSoftmaxLayer as SoftmaxLayer
from implementations.pytorch.architecture.layers.convolutional import Pytorch2DimensionalConvolutionalLayer as ConvolutionalLayer
from implementations.pytorch.architecture.layers.flatten import PytorchFlattenLayer as FlattenLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer as LinearLayer
from implementations.pytorch.architecture.layers.poolings.max_pool import Pytorch2DimensionalMaxPoolLayer as MaxPoolLayer
from implementations.pytorch.architecture.sections.convolutional import PytorchConvolutionalSection as ConvolutionalSection
from implementations.pytorch.architecture.sections.linear import PytorchLinearSection as LinearSection
from implementations.pytorch.toolbox.device import PytorchDevice as Device
from implementations.pytorch.toolbox.experiment import PytorchExperiment as Experiment
from implementations.pytorch.toolbox.data.generator import PytorchDatasetGenerator as DatasetGenerator
from implementations.pytorch.toolbox.losses.cross_entropy import PytorchCrossEntropyLossFunction as CrossEntropyLossFunction
from implementations.pytorch.toolbox.optimizers.adam import PytorchAdamOptimizer as AdamOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver as ModelSaver
from implementations.pytorch.toolbox.loader import PytorchModelLoader as ModelLoader
from implementations.pytorch.toolbox.strategies.classification import PytorchClassificationStrategy as ClassificationStrategy

# dataset = DatasetGenerator("CatDogDataset", "", 10, 42).generate(0.7, 0.2, 0.1)


architecture = (Architecture("ConvolutionalArchitecture")
                    .attach(ConvolutionalSection([
                        Block([
                            ConvolutionalLayer(in_channels=3, out_channels=33, kernel=(3, 3), stride=(2, 2), padding=(0, 0)),
                            ReLULayer(),
                            MaxPoolLayer(kernel=(5, 5), stride=(4, 4), padding=(0, 0))
                        ]),
                        Block([
                            ConvolutionalLayer(in_channels=33, out_channels=16, kernel=(3, 3), stride=(3, 3), padding=(0, 0)),
                            ReLULayer(),
                            MaxPoolLayer(kernel=(3, 3), stride=(2, 2), padding=(1, 1))
                        ])
                    ]))
                    .attach(FlattenLayer())
                    .attach(LinearSection([
                        Block([
                            LinearLayer(in_features=64, out_features=2, dimension=-1, bias=True),
                            SoftmaxLayer(n_dimensions=1)
                        ])
                    ])))

#
# experiment = Experiment(name="C3P0",
#                         architecture=architecture,
#                         optimizer=AdamOptimizer(architecture.parameters(), learning_rate=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0),
#                         loss_function=CrossEntropyLossFunction(),
#                         stopper=EarlyStopper(10, 0.01),
#                         saver=ModelSaver(""))
#
# Laboratory(name="star-wars",
#            eras=1,
#            epochs=2,
#            datagen=dataset,
#            experiments=[experiment],
#            strategy=ClassificationStrategy(),
#            logger=Logger(""),
#            loader=ModelLoader(),
#            device=Device(1)).explore()
