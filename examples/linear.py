from architecture import Architecture
from blocks.linear import LinearBlock
from discovery.tasks.training import TrainingTask
from input import Input
from pytorch.discovery.hyperparameters.losses.mse import PytorchMSELoss
from pytorch.discovery.hyperparameters.optimizers.sgd import PytorchSGD
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.linear import PytorchLinear
from pytorch.network import PytorchNetwork
from sections.linear import LinearSection


data = Input(shape=(3, 3, 3))

data = LinearSection([
    LinearBlock(PytorchLinear(1600, 120), PytorchRelu()),
    LinearBlock(PytorchLinear(120, 50), PytorchRelu()),
    LinearBlock(PytorchLinear(50, 2), PytorchRelu())])(data)

architecture = Architecture(inputs=data, network=PytorchNetwork(), name="Linear").build()
model = TrainingTask(PytorchTrainer(
    PytorchSGD(architecture.parameters(), 0.01), PytorchMSELoss())
).execute(epochs=10, architecture=architecture, training_dataset=[1, 1, 1])
