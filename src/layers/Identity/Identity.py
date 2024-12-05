from src.model.model import Layer


class Identity(Layer):
    def __init__(self):
        self.X = None

    def forward_train(self, X):
        self.X = X
        return X

    def predict(self, X):
        return X

    def backward(self, dy):
        return dy

    def updateWeights(self, optimizer, dy):
        return optimizer.updateIdentity(self, dy)