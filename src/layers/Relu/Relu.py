import numpy as np

from src.model.model import Layer


class Relu(Layer):

    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0, self.X)

    def backward(self, dy):
        return dy * (self.X > 0)
