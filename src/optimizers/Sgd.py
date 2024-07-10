from src.model.model import Optimizer


class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def updateDense(self, layer, dy):
        dX, dW, db = layer.backward(dy)
        layer.W -= self.learning_rate * dW
        layer.b -= self.learning_rate * db
        return dX

    def updateSoftmax(self, layer, dy):
        return layer.backward(dy)