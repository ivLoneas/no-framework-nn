import numpy as np

from src.model.model import Layer


class Softmax(Layer):
    def __init__(self):
        self.out = None
        self.X = None

    def forward(self, X):
        self.X = X
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # Numerical stability
        self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.out

    def backward(self, dy):
        # Reshape self.out and dy for broadcasting
        softmax_out = self.out[:, :, np.newaxis]
        dy = dy[:, np.newaxis, :]

        # Compute the Jacobian matrix efficiently
        jacobian = softmax_out * (np.eye(self.out.shape[1])[np.newaxis, :, :] - softmax_out.transpose(0, 2, 1))

        # Compute the gradient
        gradient = np.einsum('ijk,ik->ij', jacobian, dy.squeeze(axis=1))

        return gradient
