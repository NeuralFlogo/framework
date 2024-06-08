from enum import Enum

from framework.architecture.layer import Layer


class RecurrentLayer(Layer):
    def __init__(self, output_type: 'OutputType'):
        super().__init__()
        self.output = output_type

    class OutputType(Enum):
        EndSequence = 0
        HiddenStates = 1
        CellStates = 2
