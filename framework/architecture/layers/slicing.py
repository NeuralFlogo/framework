from enum import Enum

from framework.architecture.layer import Layer


class SlicingLayer(Layer):
    def __init__(self, output: 'OutputType', start: int = 0, end: int = None):
        self.output = output
        self.start = start
        self.end = end

    class OutputType(Enum):
        EndSequence = 0
        Hidden = 1
        Cell = 2