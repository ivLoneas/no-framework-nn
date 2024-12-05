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

    def updateDropout(self, layer, dy):
        return layer.backward(dy)

    def updateIdentity(self, layer, dy):
        return dy

    def updateRelu(self, layer, dy):
        return layer.backward(dy)

    def updateBatchnorm(self, layer, dy):
        dgamma, dbeta, dX = layer.backward(dy)
        layer.gamma -= self.learning_rate * dgamma
        layer.beta -= self.learning_rate * dbeta
        return dX