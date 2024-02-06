from block import Block
from layers.activation import ActivationLayer
from layers.linear import LinearLayer
from layers.recurrent import RecurrentLayer


class RecurrentBlock(Block):
    def __init__(self, recurrent: RecurrentLayer, linear: LinearLayer, activation: ActivationLayer):
        self.recurrent = recurrent
        self.linear = linear
        self.activation = activation

    def layers(self):
        return self.recurrent.get(), self.linear.get(), self.activation.get()
