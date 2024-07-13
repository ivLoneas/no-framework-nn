from abc import abstractmethod


class Layer:
    @abstractmethod
    def forward_train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def updateWeights(self, optimizer, dy):
        pass

class LossFunction:

    @abstractmethod
    def loss(self, x, y, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, x, y, *args, **kwargs):
        pass


class Optimizer:

    def __init__(self):
        self.model = None
        self.loss = None

    def step(self, dy):
        dy_tmp = dy
        for layer in self.model.layers[::-1]:
            dy_tmp = layer.updateWeights(self, dy)

    @abstractmethod
    def updateSoftmax(self, layer, dy):
        pass

    @abstractmethod
    def updateDense(self, layer, dy):
        pass
