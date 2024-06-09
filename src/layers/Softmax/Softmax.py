import numpy as np

from src.model.model import Layer


class Softmax(Layer):
    def __init__(self):
        self.out = None
        self.X = None

    def forward(self, X):
        self.X = X
        exp_X = np.exp(X - np.max(X, axis=0, keepdims=True))  # Numerical stability
        self.out = exp_X / np.sum(exp_X, axis=0, keepdims=True)
        return self.out

    def backward(self, dy):
        dX = np.zeros_like(self.X)
        for i in range(dy.shape[1]):  # Iterate over the batch
            y = self.out[:, i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            dX[:, i] = np.dot(jacobian, dy[:, i])
        return dX
