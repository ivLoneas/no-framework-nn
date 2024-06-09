from abc import abstractmethod


class Layer:
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class LossFunction:

    @abstractmethod
    def loss(self, x, y, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, x, y, *args, **kwargs):
        pass
