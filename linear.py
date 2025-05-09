import numpy as np
import math
from projet_etu import Loss, Module

# throughout this document let's keep the following notation for the input and output dimensions of any module
# input: n
# output: m

class MSELoss(Loss):
    def forward(self, y, yhat):
        # y: batch x 1{s.name for s in seq}
        # yhat: batch x 1
        # output: 1
        return np.mean((y - yhat) ** 2)
    
    def backward(self, y, yhat):
        # output: batch x 1
        return -2 * (y - yhat)
#* 0.01
class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": np.random.rand(input_dim, output_dim),
                            "b": np.zeros((1, output_dim))}
        self._gradient = {"w": None, "b": None}
        self._input = None

    def forward(self, x):
        self._input = x
        if math.isnan(self._parameters["w"][0][0]) : exit()
        return x @ self._parameters["w"] + self._parameters["b"]
    
    def backward_update_gradient(self, delta):
        # learns faster without normalization
        x = self._input
        self._gradient["w"] += x.T @ delta
        self._gradient["b"] += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, delta):
        return delta @ self._parameters["w"].T
    
    def name():
        return "Linear"