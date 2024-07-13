from src.model.model import Layer
import numpy as np


class Dense(Layer):

    def __init__(self, input_size, output_size, activation):
        self.X = None
        self.Y = None
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(self.input_size, self.output_size) * 0.01
        self.b = np.zeros(self.output_size)
        self.activation = activation

    def forward_train(self, X):
        self.X = X
        return self.activation.forward_train(np.dot(X, self.W) + self.b)

    def predict(self, X):
        return self.forward_train(X)

    def backward(self, dy):
        dy_activated = self.activation.backward(dy)
        dX = np.dot(dy_activated, self.W.T)
        dW = np.dot(self.X.T, dy_activated)
        db = np.sum(dy_activated, axis=0)
        return dX, dW, db

    def updateWeights(self, optimizer, dy):
        return optimizer.updateDense(self, dy)
