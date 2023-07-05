from blocks.pytorchNetwork import PytorchNetwork
from layers.pytorch.activation.relu import PytorchRelu
from layers.pytorch.classification.softmax import PytorchSoftMax
from layers.pytorch.convolutional.convolutional import PytorchConvolutional
from layers.pytorch.flatten.flatten import PytorchFlatten
from layers.pytorch.linear.linear import PytorchLinear
from layers.pytorch.pool.max import PytorchMaxPooling

layers = [
    PytorchConvolutional(1, 6, 5),
    PytorchMaxPooling(),
    PytorchRelu(),
    PytorchConvolutional(7, 8, 5),
    PytorchMaxPooling(),
    PytorchRelu(),
    PytorchFlatten(1, 3),
    PytorchLinear(1600, 120),
    PytorchRelu(),
    PytorchLinear(1600, 120),
    PytorchSoftMax(2)
]

model = PytorchNetwork(layers)

print(model.parameters())
