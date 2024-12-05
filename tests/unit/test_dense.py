import unittest

import numpy as np

from src.layers.Dense.Dense import Dense
from src.layers.Identity.Identity import Identity


class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        self.input_size = 4
        self.output_size = 3
        self.activation = Identity()
        self.dense_layer = Dense(self.input_size, self.output_size, self.activation)
        self.X = np.random.randn(5, self.input_size)  # Example input
        self.dy = np.random.randn(5, self.output_size)  # Example gradient for backward pass

    def test_forward(self):
        # Perform forward pass
        output = self.dense_layer.forward_train(self.X)
        self.assertEqual(output.shape, (5, self.output_size), "Forward pass output shape is incorrect")

    def test_backward(self):
        # Perform forward pass first
        self.dense_layer.forward_train(self.X)
        # Perform backward pass
        dX, dW, db = self.dense_layer.backward(self.dy)
        self.assertEqual(dX.shape, (5, self.input_size), "Backward pass dX shape is incorrect")
        self.assertEqual(dW.shape, (self.input_size, self.output_size), "Backward pass dW shape is incorrect")
        self.assertEqual(db.shape, (self.output_size,), "Backward pass db shape is incorrect")

        # Check if the gradients are not None
        self.assertIsNotNone(dX, "dX should not be None")
        self.assertIsNotNone(dW, "dW should not be None")
        self.assertIsNotNone(db, "db should not be None")


if __name__ == '__main__':
    unittest.main()
