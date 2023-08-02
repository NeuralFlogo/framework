class Input:
    def __init__(self, shape):
        self.shape = shape
        self.trail = []

    def add(self, step):
        self.trail.append(step)
        return self
