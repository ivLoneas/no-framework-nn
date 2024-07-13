import numpy as np

from src.model.model import Layer


class Relu(Layer):

    def __init__(self):
        self.X = None

    def forward_train(self, X):
        self.X = X
        return np.maximum(0, self.X)

    def predict(self, X):
        return self.forward_train(X)

    def backward(self, dy):
        return dy * (self.X > 0)

    def updateWeights(self, optimizer, dy):
        return optimizer.updateRelu(self, dy)
