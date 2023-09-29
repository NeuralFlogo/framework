from architecture import Architecture
from blocks.recurrent import RecurrentBlock
from pytorch.layers.activations.softmax import PytorchSoftmax
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.recurrent.gru import PytorchGRU
from pytorch.layers.recurrent.lstm import PytorchLSTM
from pytorch.layers.recurrent.rnn import PytorchRNN
from pytorch.pmodel import PytorchModel
from sections.recurrent import RecurrentSection

rnn_architecture = Architecture(PytorchModel(), "rnn") \
    .attach(RecurrentSection(RecurrentBlock(PytorchRNN(50, 12, 4), PytorchLinear(100, 50), PytorchSoftmax(5)))).build()

gru_architecture = Architecture(PytorchModel(), "gru") \
    .attach(RecurrentSection(RecurrentBlock(PytorchGRU(50, 12, 4), PytorchLinear(100, 50), PytorchSoftmax(5)))).build()

lstm_architecture = Architecture(PytorchModel(), "lstm") \
    .attach(RecurrentSection(RecurrentBlock(PytorchLSTM(50, 12, 4), PytorchLinear(100, 50), PytorchSoftmax(5)))).build()

print(rnn_architecture)
