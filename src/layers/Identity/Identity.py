from src.model.model import Layer


class Identity(Layer):
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return X

    def backward(self, dy):
        return dy
