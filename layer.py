class Layer:
    def __call__(self, x):
        return x.add(self)
