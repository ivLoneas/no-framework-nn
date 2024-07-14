import numpy as np

from src.model.model import Layer


class Batchnorm(Layer):

    def __init__(self, input_size, momentum = 0.9):
        self.gamma = np.random.randn(input_size)
        self.beta = np.zeros(input_size)
        self.momentum = momentum
        self.epsilon = 1e-5
        self.running_mu = np.zeros(input_size)
        self.running_sigma = np.zeros(input_size)
        self.N = 0

    def forward_train(self, X):
        self.batch_size = X.shape[0]
        self.X = X
        self.mu = np.mean(X, axis = 0)
        self.sigma = np.std(X, axis = 0) + self.epsilon
        self.X_norm = (X - self.mu) / self.sigma
        self.output = self.gamma * self.X_norm + self.beta
        self.running_mu = self.running_mu * self.momentum + (1-self.momentum) * self.mu
        self.running_sigma = self.running_sigma * self.momentum + (1-self.momentum) * self.sigma
        self.N += 1
        return self.output

    def predict(self, X):
        X_norm = (X - self.running_mu) / self.running_sigma
        return self.gamma * X_norm + self.beta

    def backward(self, dy):
        dgamma = np.sum(dy * self.X_norm, axis=0)
        dbeta = np.sum(dy, axis=0)
        # Gradient of the normalized input
        dX_norm = dy * self.gamma

        # Gradient of the input
        dsigma = np.sum(dX_norm * (self.X - self.mu) * -0.5 * (self.sigma**-3), axis=0)
        dmu = np.sum(dX_norm * -1 / self.sigma, axis=0) + dsigma * np.mean(-2 * (self.X - self.mu), axis=0)
        dX = dX_norm / self.sigma + dsigma * 2 * (self.X - self.mu) / self.batch_size + dmu / self.batch_size
        return dgamma, dbeta, dX

    def updateWeights(self, optimizer, dy):
        return optimizer.updateBatchnorm(self, dy)