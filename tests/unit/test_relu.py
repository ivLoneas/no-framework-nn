import unittest

import numpy as np
from numpy.testing import assert_array_equal
from src.layers.Relu.Relu import Relu


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.relu = Relu()

    def testForward(self):
        x1 = np.array([[1, -1]])
        x2 = np.array([[1, 1, 1],
                       [1, 1, 1]])
        output1 = self.relu.forward_train(x1)
        output2 = self.relu.forward_train(x2)

        assert_array_equal(output1, np.array([[1, 0]]), "Forward pass should have no negative values")
        assert_array_equal(output2, x2, "Forward pass output on regular case should not have positive values incorrect")

        self.assertEqual(output1.shape, (1, 2), "Forward pass output shape is incorrect for 1 vector")
        self.assertEqual(output2.shape, (2, 3), "Forward pass output shape is incorrect for regular case")

    def testBackward(self):
        x1 = np.array([[1, -1]])
        self.relu.forward_train(x1)
        dy1 = np.array([[1, 1]])
        output1 = self.relu.backward(dy1)

        assert_array_equal(output1, np.array([[1, 0]]), "Backward pass 1 vector incorrect")
        self.assertEqual(output1.shape, (1, 2), "Backwards pass output shape incorrect")


if __name__ == '__main__':
    unittest.main()
