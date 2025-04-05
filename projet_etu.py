import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class Module(object):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": np.random.randn(input_dim, output_dim),
                            "b": np.zeros((1, output_dim))}
        self._gradient = {"w": None, "b": None}
        self._input = None
        self._output = None

    def __str__(self):
        return f'Weight: {self._parameters["w"]}\nBias: {self._parameters["b"]}'

    def zero_grad(self):
        self._gradient["w"] = np.zeros_like(self._parameters["w"])
        self._gradient["b"] = np.zeros_like(self._parameters["b"])

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-5):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters["w"] -= gradient_step*self._gradient["w"]
        self._parameters["b"] -= gradient_step*self._gradient["b"]

    def backward_update_gradient(self, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass