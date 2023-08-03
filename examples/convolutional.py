from blocks.convolutional import ConvolutionalBlock
from blocks.linear import LinearBlock
from architecture import Architecture
from discovery.tasks.training import TrainingTask
from input import Input
from pytorch.discovery.hyperparameters.losses.mse import PytorchMSELoss
from pytorch.discovery.hyperparameters.optimizers.sgd import PytorchSGD
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.activations.softmax import PytorchSoftMax
from pytorch.layers.convolutional import PytorchConvolutional
from pytorch.layers.flatten import PytorchFlatten
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.poolings.max import PytorchMaxPooling
from pytorch.network import PytorchNetwork
from sections.convolutional import ConvolutionalSection
from sections.linear import LinearSection

data = Input(shape=(6, 6, 6))
data = ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(1, 6, 5), PytorchRelu())], PytorchMaxPooling())(data)
data = ConvolutionalSection([ConvolutionalBlock(PytorchConvolutional(7, 8, 5), PytorchRelu())], PytorchMaxPooling())(data)
data = PytorchFlatten(1, 3)(data)
outputs = LinearSection([LinearBlock(PytorchLinear(1600, 120), PytorchRelu()), LinearBlock(PytorchLinear(120, 2), PytorchSoftMax(2))])(data)

architecture = Architecture(inputs=data, network=PytorchNetwork(), name="Convolutional").build()
model = TrainingTask(PytorchTrainer(
    PytorchSGD(architecture.parameters(), 0.01), PytorchMSELoss())
).execute(epochs=10, architecture=architecture, training_dataset=[1, 1, 1])
