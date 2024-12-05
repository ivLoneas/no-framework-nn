import numpy as np


def accuracy(x, y):
    return np.sum((x * y)) / y.shape[0]
