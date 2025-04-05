import numpy as np
from projet_etu import Loss, Module

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
    
    def backward(self,delta):#the delta its the result of the loss_fun.backward
        seq = self.seq.copy()
        delta_t = delta
        for s in seq[1:][::-1]: #we want the inverse way because we are doing it backwards
            s.backward_update_gradient(delta_t)
            delta_t = s.backward_delta(delta_t)

        #this is the last/fist module
        s.backward_update_gradient(delta_t)


    def update_parameters(self):
        for s in self.seq :
            s.update_parameters()
        



    

    
    