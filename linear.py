import numpy as np
from projet_etu import Loss, Module

# throughout this document let's keep the following notation for the input and output dimensions of any module
# input: n
# output: m

class MSELoss(Loss):
    def forward(self, y, yhat):
        # y: batch x 1
        # yhat: batch x 1
        # output: 1
        return np.mean((y - yhat) ** 2)
    
    def backward(self, y, yhat):
        # output: batch x 1
        return -2 * (y - yhat)

class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": np.random.randn(input_dim, output_dim) * 0.01,
                            "b": np.zeros((1, output_dim))}
        self._gradient = {"w": None, "b": None}
        self._input = None

    def zero_grad(self):
        self._gradient["w"] = np.zeros_like(self._parameters["w"])
        self._gradient["b"] = np.zeros_like(self._parameters["b"])

    def forward(self, x):
        self._input = x
        return x @ self._parameters["w"] + self._parameters["b"]
    
    def backward_update_gradient(self, delta):
        # ADD SOME NORMALIZATION
        x = self._input
        self._gradient["w"] += x.T @ delta
        self._gradient["b"] += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, delta):
        return delta @ self._parameters["w"].T