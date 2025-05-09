import numpy as np

# File contains declarations of Modules
# And also other useful constructions, e.g. Seq, Optim
#--------------------------------------------------
# Base Module Class
class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def __str__(self):
        return f'Weight: {self._parameters["w"]}\nBias: {self._parameters["b"]}'

    def zero_grad(self):
        self._gradient["w"] = np.zeros_like(self._parameters["w"])
        self._gradient["b"] = np.zeros_like(self._parameters["b"])

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters["w"] -= gradient_step*self._gradient["w"]
        self._parameters["b"] -= gradient_step*self._gradient["b"]

    def backward_update_gradient(self, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

    def name(self):
        pass

#--------------------------------------------------
# Linear Module
class Linear(Module):
    def __init__(self, input_dim, output_dim):
        self._parameters = {"w": np.random.randn(input_dim, output_dim),
                            "b": np.random.randn(1, output_dim)}
        self._gradient = {"w": np.zeros((input_dim, output_dim)), "b": np.zeros((1, output_dim))}
        self._input = None

    def forward(self, x):
        self._input = x
        return x @ self._parameters["w"] + self._parameters["b"]
    
    def backward_update_gradient(self, delta):
        x = self._input
        self._gradient["w"] += x.T @ delta
        self._gradient["b"] += np.sum(delta, axis=0, keepdims=True)

    def backward_delta(self, delta):
        return delta @ self._parameters["w"].T
    
    def name():
        return "Linear"
    
#--------------------------------------------------
# Hyperbolic Tangent Module
class TanH(Module):
    def forward(self, x):
        self._output = np.tanh(x)
        return self._output
    
    # re-define functions for the Sequential Network
    # this way, nothing happens for this module
    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, delta):
        pass
    
    def backward_delta(self, delta):
        return (1 - self._output ** 2) * delta
    
    def name():
        return "TanH"
    
#--------------------------------------------------
# Sigmoid Module
class Sigmoide(Module):
    def forward(self, x):
        self._output = 1 / (1 + np.exp(-x))
        return self._output

    # re-define functions for the Sequential Network
    # this way, nothing happens for this module
    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, delta):
        pass

    def backward_delta(self, delta):
        return self._output * (1 - self._output) * delta
    
    def name():
        return "Sigmoide"
    
#--------------------------------------------------
# Softmax Module
class Softmax(Module):
    def forward(self, x):
        # with one-hot encoding of labels
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        sum_x = np.sum(exp_x, axis=1, keepdims=True)
        self._output = exp_x / sum_x
        return self._output
    
    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass
    
    def backward_update_gradient(self, delta):
        pass

    def backward_delta(self, delta):
        return self._output * (1 - self._output) * delta
    
    def name():
        return "Softmax"

#--------------------------------------------------
# Sequential Class for wrapping together multiple modules
class Sequential(object):
    def __init__(self,sequence):
        if not all(isinstance(func, Module) for func in sequence):
            raise ValueError("All modules must be of type 'Module'")
        self.seq = sequence

    def __str__(self):
        return f'The sequence of the following modules {", ".join(s.name for s in self.seq)}'
    
    def zero_grad(self):
        for s in self.seq:
            s.zero_grad()

    def forward(self,x):
        x_next = x
        for s in self.seq:
            x_next = s.forward(x_next)
        return x_next #this will be the y_hat of the end of the sequence
    
    def backward(self,delta): #the delta its the result of the loss_fun.backward
        seq = self.seq.copy()
        delta_t = delta
        for s in seq[1:][::-1]: #we want the inverse way because we are doing it backwards
            s.backward_update_gradient(delta_t)
            delta_t = s.backward_delta(delta_t)

        #this is the last/first module
        s.backward_update_gradient(delta_t)

    def update_parameters(self, gradient_step):
        for s in self.seq :
            s.update_parameters(gradient_step)
        
#--------------------------------------------------
# Optim class for encapsulating the execution of GD
class Optim(object):
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    # execute one step of GD
    def step(self, batch_x, batch_y, file):
        self.net.zero_grad()
        y_pred = self.net.forward(batch_x)
        file.write(f"predicted:\n{y_pred}\n")

        loss = self.loss.forward(batch_y, y_pred)
        file.write(f"loss:\n{loss}\n")

        delta = self.loss.backward(batch_y, y_pred)
        file.write(f"delta:\n{delta}\n")

        self.net.backward(delta)
        self.net.update_parameters(self.eps)

        return loss
    
    # run the whole GD automatically using batches
    def SGD(self, x, y, batch_size, n_epochs, verbose = True):
        losses = []

        with open("test.txt", "w") as file:
            for epoch in range(n_epochs):
                file.write(f"EPOCH {epoch}\n")

                # shuffle dataset at the start of each epoch
                order = np.random.permutation(np.arange(x.shape[0]))
                x = x[order]
                y = y[order]

                for i in range(0, x.shape[0], batch_size):
                    file.write(f"ITERATION {i}\n")
                    x_batch = x[i : i + batch_size]
                    y_batch = y[i : i + batch_size]
                    file.write(f"x_batch:\n{x_batch}\n")
                    file.write(f"y_batch:\n{y_batch}\n")

                    loss = self.step(x_batch, y_batch, file)
                    losses.append(loss)

                    file.write("\n")
                file.write("\n")
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss = {losses[-1]}")

        return losses