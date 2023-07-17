class Input:
    def __init__(self, shape):
        self.shape = shape
        self.route = []

    def add(self, component):
        self.route.append(component)
        return self
