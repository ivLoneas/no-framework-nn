import numpy as np

from src.model.model import LossFunction


class CrossEntropyError(LossFunction):
    @staticmethod
    def loss(p, y, *args, **kwargs):
        epsilon = 1e-12  # Small constant to prevent log(0)
        p = np.clip(p, epsilon, 1. - epsilon)
        return -np.sum(y * np.log(p)) / y.shape[0]

    @staticmethod
    def gradient(p, y, *args, **kwargs):
        epsilon = 1e-12
        p = np.clip(p, epsilon, 1. - epsilon)
        return (-y / p) / y.shape[0]
