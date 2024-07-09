import unittest

import numpy as np

from src.train.batch_index_generator import ButchIndexGenerator


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.n_size = 10
        self.batch_size = 3

    def test_iterator_size(self):
        ig = ButchIndexGenerator(self.n_size, self.batch_size)
        self.assertEqual(3, len(ig.next()))
        self.assertEqual(3, len(ig.next()))
        self.assertEqual(3, len(ig.next()))
        self.assertEqual(1, len(ig.next()))

    def test_iterator_unique_values(self):
        ig = ButchIndexGenerator(self.n_size, self.batch_size)
        values = [idx for _ in range(4) for idx in ig.next()]
        self.assertEqual(10, len(np.unique(values)))

if __name__ == '__main__':
    unittest.main()
