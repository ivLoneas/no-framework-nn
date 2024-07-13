import unittest

import numpy as np

from src.layers.Dropout.Dropout import Dropout


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.dropout = Dropout()

    def testForward(self):
        # TODO. No idea what tests for dropout
        x1 = np.array([[1, 1]])



if __name__ == '__main__':
    unittest.main()
