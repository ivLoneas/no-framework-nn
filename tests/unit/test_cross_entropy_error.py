import unittest

import numpy as np
from numpy.testing import assert_array_equal

from src.errors.Cee.CrossEntropyError import CrossEntropyError


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.cee = CrossEntropyError()

    def test_loss(self):
        y1 = np.array([[0, 1, 0]])
        x1 = np.array([[0.1, 0.8, 0.1]])

        self.assertEqual(-np.log(0.8), self.cee.loss(x1, y1), "Loss calculated wrong")  # add assertion here

        y2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        x2 = np.array([[0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25]])

        self.assertEqual(-np.log(0.25), self.cee.loss(x2, y2), "Loss calculated wrong")

    def test_backward(self):
        y1 = np.array([[0, 1, 0]])
        x1 = np.array([[0.1, 0.8, 0.1]])

        loss = self.cee.loss(x1, y1)
        assert_array_equal(np.array([[0, -1 / 0.8, 0]]), self.cee.gradient(x1, y1), "Wrong gradient")

        y2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        x2 = np.array([[0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25]])

        loss = self.cee.loss(x2, y2)
        assert_array_equal(np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
                           self.cee.gradient(x2, y2), "Wrong gradient")


if __name__ == '__main__':
    unittest.main()
