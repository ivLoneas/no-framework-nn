import unittest

import numpy as np
from numpy.testing import assert_array_equal

from src.errors.Mse.Mse import Mse


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.mse = Mse()

    def test_loss(self):
        x1 = np.array([[1, 2, 2]])
        y1 = np.array([[0, 0, 0]])
        loss1 = self.mse.loss(x1, y1)
        zero_loss1 = self.mse.loss(x1, x1)
        self.assertEqual(3.0, loss1, "MSE loss calculated wrong")
        self.assertEqual(0, zero_loss1, "MSE loss calculated wrong for exact solution")

        x2 = np.array([1, 2, 2])
        y2 = np.array([0, 0, 0])
        loss2 = self.mse.loss(x2, y2)
        zero_loss2 = self.mse.loss(x2, x2)
        self.assertEqual(3.0, loss2, "MSE loss calculated wrong for 1d")
        self.assertEqual(0, zero_loss2, "MSE loss calculated wrong for exact solution for 1d")

    def test_gradient(self):
        x1 = np.array([[1, 2, 2]])
        y1 = np.array([[0, 0, 0]])
        grad1 = self.mse.gradient(x1, y1)
        zero_grad = self.mse.gradient(x1, x1)

        assert_array_equal(np.array([[2, 4, 4]]) / 3, grad1, "Gradient calculated wrong")
        assert_array_equal(np.array([[0, 0, 0]]), zero_grad, "Gradient calculated wrong for exact solution")

        x2 = np.array([[1, 2, 2]])
        y2 = np.array([[0, 0, 0]])
        grad2 = self.mse.gradient(x2, y2)
        zero_grad2 = self.mse.gradient(x2, x2)

        assert_array_equal(np.array([[2, 4, 4]]) / 3, grad2, "Gradient calculated wrong for 1d")
        assert_array_equal(np.array([[0, 0, 0]]), zero_grad2, "Gradient calculated wrong for exact solution for 1d")


if __name__ == '__main__':
    unittest.main()
