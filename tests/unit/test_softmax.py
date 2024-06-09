import unittest

import numpy as np
from numpy.testing import assert_array_equal

from src.layers.Softmax.Softmax import Softmax


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.softmax = Softmax()
        self.e = np.exp(1)

    def testForward(self):
        x1 = np.array([[1, 1, 1]])
        output1 = self.softmax.forward(x1)

        assert_array_equal(np.array([[1, 1, 1]]) / 3, output1, "Wrong forward output for one element")

        x2 = np.array([[1, 2, 3], [4, 5, 6]])
        output2 = self.softmax.forward(x2)
        res2 = np.array([[self.e ** (-2), self.e ** (-1), 1] / np.sum([self.e ** (-2), self.e ** (-1), 1]),
                         [self.e ** (-2), self.e ** (-1), 1] / np.sum([self.e ** (-2), self.e ** (-1), 1])])
        assert_array_equal(res2, output2, "Wrong forward output for regular case")

        if __name__ == '__main__':
            unittest.main()
