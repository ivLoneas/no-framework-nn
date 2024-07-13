import numpy as np


class BatchIndexGenerator:
    def __init__(self, n_size, batch_size):
        self.n_size = n_size
        self.batch_size = batch_size
        self.g = self.generator()

    def generator(self):
        ordered = np.arange(self.n_size)
        while True:
            indexes = np.random.permutation(ordered)
            for start in range(0, self.n_size, self.batch_size):
                end = start + self.batch_size
                yield indexes[start:end]

    def next(self):
        return next(self.g)
