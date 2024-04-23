from framework.toolbox.laboratory import Laboratory
from framework.toolbox.logger import Logger
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture as Architecture
from implementations.pytorch.architecture.block import PytorchBlock as Block
from implementations.pytorch.architecture.layers.activations.softmax import PytorchSoftmaxLayer as SoftmaxLayer
from implementations.pytorch.architecture.layers.linear import PytorchLinearLayer as LinearLayer
from implementations.pytorch.architecture.layers.recurrents.rnn import PytorchRNNLayer as RNNLayer
from implementations.pytorch.architecture.layers.slicing import PytorchSlicingLayer as SlicingLayer
from implementations.pytorch.architecture.sections.recurrent import PytorchRecurrentSection as RecurrentSection
from implementations.pytorch.toolbox.device import PytorchDevice as Device
from implementations.pytorch.toolbox.experiment import PytorchExperiment as Experiment
from implementations.pytorch.toolbox.data.generator import PytorchDatasetGenerator
from implementations.pytorch.toolbox.losses.cross_entropy import PytorchCrossEntropyLossFunction as CrossEntropyLossFunction
from implementations.pytorch.toolbox.optimizers.sgd import PytorchSGDOptimizer as SGDOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver as ModelSaver
from implementations.pytorch.toolbox.loader import PytorchModelLoader as ModelLoader
from implementations.pytorch.toolbox.strategies.classification import PytorchClassificationStrategy as ClassificationStrategy

dataset = PytorchDatasetGenerator("Mnist", "", 10, 42).generate(0.7, 0.2, 0.1)

architecture = (Architecture("RecurrentArchitecture")
                    .attach(RecurrentSection([
                        Block([
                            RNNLayer(input_size=784, hidden_size=100, num_layer=1, bidirectional=False, dropout=0),
                            SlicingLayer(output=SlicingLayer.OutputType.EndSequence, start=0, end=101),
                            LinearLayer(in_features=100, out_features=10, dimension=-1, bias=True),
                            SoftmaxLayer(n_dimensions=1)
                        ])
                    ])))

experiment = Experiment(name="R9H3",
                        architecture=architecture,
                        optimizer=SGDOptimizer(architecture.parameters(), learning_rate=0.05, momentum=0, dampening=0, weight_decay=0),
                        loss_function=CrossEntropyLossFunction(),
                        stopper=EarlyStopper(10, 0.01),
                        saver=ModelSaver(""))

Laboratory(name="star-wars",
           eras=1,
           epochs=2,
           datagen=dataset,
           experiments=[experiment],
           strategy=ClassificationStrategy(),
           logger=Logger(""),
           loader=ModelLoader(),
           device=Device(1)).explore()