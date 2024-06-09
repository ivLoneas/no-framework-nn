import unittest

import numpy as np
from numpy.testing import assert_array_equal

from src.layers.Mse.Mse import Mse


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.mse = Mse()

    def test_loss(self):
        x1 = np.array([[1, 2, 2]])
        y1 = np.array([[0, 0, 0]])
        loss1 = self.mse.loss(x1, y1)
        zero_loss = self.mse.loss(x1, x1)
        self.assertEqual(3.0, loss1, "MSE loss calculated wrong")
        self.assertEqual(0, zero_loss, "MSE loss calculated wrong for exact solution")

    def test_gradient(self):
        x1 = np.array([[1, 2, 2]])
        y1 = np.array([[0, 0, 0]])
        grad1 = self.mse.gradient(x1, y1)
        zero_grad = self.mse.gradient(x1, x1)

        assert_array_equal(np.array([[2, 4, 4]]) / 3, grad1, "Gradient calculated wrong")
        assert_array_equal(np.array([[0, 0,0 ]]), zero_grad, "Gradient calculated wrong for exact solution")


if __name__ == '__main__':
    unittest.main()
