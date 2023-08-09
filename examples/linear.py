import torch

from architecture import Architecture
from framework.structure.blocks.linear import LinearBlock
from framework.data.dataset.dataset import Dataset
from framework.data.dataset.entry import Entry
from framework.discovery.tasks.training import TrainingTask
from input import Input
from pytorch.discovery.hyperparameters.losses.mse import PytorchMSELoss
from pytorch.discovery.hyperparameters.optimizers.sgd import PytorchSGD
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.layers.activations.relu import PytorchRelu
from pytorch.layers.linear import PytorchLinear
from pytorch.layers.normalizations.dropout import PytorchDropout
from pytorch.network import PytorchNetwork
from framework.structure.sections.linear import LinearSection


entry1 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry2 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry3 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry4 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry5 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry6 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry7 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry8 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry9 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))
entry10 = Entry(1, torch.Tensor([20, 0]), torch.Tensor([60]))

dataset = Dataset([entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry8, entry9, entry10])

data = Input(shape=(4, 4, 4))

x = LinearSection([
    LinearBlock(PytorchLinear(2, 120), PytorchRelu(), PytorchDropout(0.01)),
    LinearBlock(PytorchLinear(120, 50), PytorchRelu(), PytorchDropout(0.01)),
    LinearBlock(PytorchLinear(50, 2), PytorchRelu(), PytorchDropout(0.01))])(data)

x = LinearSection([
    LinearBlock(PytorchLinear(2, 120), PytorchRelu(), PytorchDropout(0.01)),
    LinearBlock(PytorchLinear(120, 50), PytorchRelu(), PytorchDropout(0.01)),
    LinearBlock(PytorchLinear(50, 2), PytorchRelu(), PytorchDropout(0.01))])(x)

x = LinearSection([
    LinearBlock(PytorchLinear(2, 120), PytorchRelu(), PytorchDropout(0.01)),
    LinearBlock(PytorchLinear(120, 50), PytorchRelu(), PytorchDropout(0.01)),
    LinearBlock(PytorchLinear(50, 2), PytorchRelu(), PytorchDropout(0.01))])(x)

architecture = Architecture(input=x, network=PytorchNetwork(), name="Linear").build()


model = TrainingTask(PytorchTrainer(PytorchSGD(architecture.parameters(), 0.01), PytorchMSELoss()))\
    .execute(epochs=10, architecture=architecture, training_dataset=dataset)
