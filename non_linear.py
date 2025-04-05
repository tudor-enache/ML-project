import numpy as np
from projet_etu import Module

# nothing to learn for the modules here, since they serve as ACTIVATIONs

class TanH(Module):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": np.zeros((input_dim, output_dim)),
                            "b": np.zeros((1, output_dim))}
        self._gradient = {"w": None, "b": None}
        self._input = None

    def forward(self, x):
        self._input = x
        self._output = np.tanh(x @ self._parameters["w"] + self._parameters["b"])
        return self._output
    
    def backward_delta(self, delta):
        return (1 - self._output ** 2) * delta
    
class Sigmoide(Module):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": np.zeros((input_dim, output_dim)),
                            "b": np.zeros((1, output_dim))}
        self._gradient = {"w": None, "b": None}
        self._input = None

    def forward(self, x):
        self._input = x
        self._output = 1 / (1 + np.exp(-x))
        return self._output
    
    def backward_delta(self, delta):
        return self._output * (1 - self._output) * delta