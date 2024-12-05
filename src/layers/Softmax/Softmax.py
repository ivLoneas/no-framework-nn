import numpy as np

from src.model.model import Layer


class Softmax(Layer):
    def __init__(self):
        self.out = None
        self.X = None

    def forward_train(self, X):
        self.X = X
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # Numerical stability
        self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.out

    def predict(self, X):
        soft = self.forward_train(X)
        max_indices = np.argmax(soft, axis=1)
        # Create a zero matrix of the same shape as softmax_outputs
        one_hot_outputs = np.zeros_like(soft)
        # Set the appropriate elements to 1
        one_hot_outputs[np.arange(len(soft)), max_indices] = 1
        return one_hot_outputs

    def backward(self, dy):
        # Reshape self.out and dy for broadcasting
        softmax_out = self.out[:, :, np.newaxis]
        dy = dy[:, np.newaxis, :]

        # Compute the Jacobian matrix efficiently
        jacobian = softmax_out * (np.eye(self.out.shape[1])[np.newaxis, :, :] - softmax_out.transpose(0, 2, 1))

        # Compute the gradient
        gradient = np.einsum('ijk,ik->ij', jacobian, dy.squeeze(axis=1))

        return gradient

    def updateWeights(self, optimizer, dy):
        return optimizer.updateSoftmax(self, dy)