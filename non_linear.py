import numpy as np
from projet_etu import Module

# nothing to learn for the modules here, since they serve as ACTIVATIONs

class TanH(Module):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": None,
                            "b": None}
        self._gradient = {"w": None, "b": None}
        self._input = None

    def forward(self, x):
        self._input = x
        #self._output = np.tanh(x @ self._parameters["w"] + self._parameters["b"])
        return np.tanh(x)
    
    def update_parameters(self, gradient_step=1e-5):
        pass

    def backward_update_gradient(self, delta):
        return delta
    
    def backward_delta(self, delta):
        return (1 - self._input ** 2) * delta
    
    def name():
        return "TanH"
    
class Sigmoide(Module):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": None,
                            "b": None}
        self._gradient = {"w": None, "b": None}
        self._input = None

    def forward(self, x):
        self._input = x
        self._output = 1 / (1 + np.exp(-x))
        return self._output
    
    def update_parameters(self, gradient_step=1e-5):
        pass

    def backward_update_gradient(self, delta):
        return delta

    def backward_delta(self, delta):
        return self._input * (1 - self._input) * delta
    
    def name():
        return "Sigmoide"