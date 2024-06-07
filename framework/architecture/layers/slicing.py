from framework.architecture.layer import Layer


class SlicingLayer(Layer):
    def __init__(self, start: int = 0, end: int = None):
        self.start = start
        self.end = end
