import numpy as np

from src.model.model import LossFunction


class Mse(LossFunction):

    def loss(self, x, y, *args, **kwargs):
        return np.mean((x - y) ** 2) / np.size(y)

    def gradient(self, x, y, *args, **kwargs):
        return 2 * (x - y) / np.size(y)
