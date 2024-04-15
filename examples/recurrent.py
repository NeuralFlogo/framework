import torch

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
from implementations.pytorch.toolbox.experiment import PytorchExperiment as Experiment
from implementations.pytorch.toolbox.generator import PytorchDatasetGenerator
from implementations.pytorch.toolbox.losses.cross_entropy import \
    PytorchCrossEntropyLossFunction as CrossEntropyLossFunction
from implementations.pytorch.toolbox.optimizers.sgd import PytorchSGDOptimizer as SGDOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver as ModelSaver
from implementations.pytorch.toolbox.strategies.classification import \
    PytorchClassificationStrategy as ClassificationStrategy

PATH = "C:/Users/juanc/Downloads/digit-recognizer/train.csv"
DATASET_NAME = ""
dataset = PytorchDatasetGenerator(DATASET_NAME, PATH, 10, 42).generate(0.7, 0.2, 0.1)

architecture = (Architecture("RecurrentArchitecture")
                    .attach(RecurrentSection([
                        Block([
                            RNNLayer(input_size=784, hidden_size=100, num_layer=1, bidirectional=False, dropout=0),
                            SlicingLayer(output=SlicingLayer.OutputType.EndSequence, start=0, end=101),
                            LinearLayer(in_features=100, out_features=10, dimension=-1, bias=True),
                            SoftmaxLayer(n_dimensions=1)
                        ])
                    ])))

experiment = Experiment("R9H3",
                               SGDOptimizer(architecture.parameters(), learning_rate=0.05, momentum=0, dampening=0, weight_decay=0),
                               CrossEntropyLossFunction(),
                               EarlyStopper(10, 0.01),
                               ModelSaver("C:/Users/juanc/Downloads/test/"))

_, model = Laboratory("star-wars",
                      1,
                      dataset,
                      architecture,
                      [experiment],
                      ClassificationStrategy(),
                      Logger("C:/Users/juanc/Downloads/test/log.txt"), 1).explore()

print(model.predict(torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,17,17,
                                   17,17,81,180,180,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,253,253,253,253,253,
                                   253,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,228,253,253,253,253,253,253,253,207,197,
                                   46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,213,253,253,253,253,253,253,253,253,253,253,223,52,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,231,253,253,253,108,40,40,115,244,253,253,134,3,0,0,0,0,0
                                      ,0,0,0,0,0,0,0,0,0,0,63,114,114,114,37,0,0,0,205,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,57,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,
                                   253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,253,253,253,15,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,205,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,
                                   99,96,0,0,45,224,253,253,195,10,0,0,0,0,0,0,0,0,0,0,0,11,25,105,83,189,189,228,253,251,189,
                                   189,218,253,253,210,27,0,0,0,0,0,0,0,0,0,0,42,116,173,253,253,253,253,253,253,253,253,253,
                                   253,253,253,253,221,116,7,0,0,0,0,0,0,0,0,0,118,253,253,253,253,245,212,222,253,253,253,253,
                                   253,253,253,253,253,253,160,15,0,0,0,0,0,0,0,0,254,253,253,253,189,99,0,32,202,253,253,253,
                                   240,122,122,190,253,253,253,174,0,0,0,0,0,0,0,0,255,253,253,253,238,222,222,222,241,253,253,
                                   230,70,0,0,17,175,229,253,253,0,0,0,0,0,0,0,0,158,253,253,253,253,253,253,253,253,205,106,65
                                      ,0,0,0,0,0,62,244,157,0,0,0,0,0,0,0,0,6,26,179,179,179,179,179,30,15,10,0,0,0,0,0,0,0,0,14,
                                   6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=torch.float32)).argmax(1))