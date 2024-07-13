import numpy as np
from src.model.model import Layer


class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward_train(self, X):
        self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
        return X * self.mask

    def predict(self, X):
        return X

    def backward(self, dY):
        return dY * self.mask

    def updateWeights(self, optimizer, dy):
        return optimizer.updateDropout(self, dy)