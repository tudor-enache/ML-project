import numpy as np
from projet_etu import Loss, Module

class MSELoss(Loss):
    def forward(self, y, yhat):
        # y, yhat: batch x d
        # output: batch
        return ((y - yhat) ** 2).sum(axis=1)
    
    def backward(self, y, yhat):
        return -2 * (y - yhat)

class Linear(Module):
    def __init__(self, input, output):
        self._parameters = {"input": input, "output": output, "w": np.random.randn(input, output)}

    def zero_grad(self):
        self._gradient = None

    def forward(self, x):
        w = self._parameters["w"]
        return x @ w
    
    def backward_update_gradient(self, input, delta):
        pass