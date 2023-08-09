class Input:
    def __init__(self, shape):
        self.shape = shape
        self.route = []

    def add_layer(self, component):
        self.route.append(component)
        return self

    def add(self, other):
        self.__sum_shape(other.shape)
        return self

    def __sum_shape(self, shape):
        self.shape = self.shape





