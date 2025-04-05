import numpy as np
from projet_etu import Loss, Module
from sequential import *

class Optim(object):
    def __init__(self,net,loss_fn,eps):
        if isinstance(net, Sequential) and isinstance(loss_fn, Module) :
            raise ValueError("Input have wrong type of data")
        self.net = net
        self.loss_fn = loss_fn
        self.eps = eps

    def __str__(self):
        return 'optimization of a given network with a loss function and an epsilon'
    
    def step(self,batch_x,batch_y):
        y_hat = self.net.forward(batch_x)

        self.loss_fn.forward(batch_y, y_hat)

        self.net.zero_grad()

        delta = self.loss_fn.backward(batch_y, y_hat)
        self.net.backward(delta)


        self.net.update_parameters()
    
    def SGD(self,net2,x,y,batch_nb,num_epochs):

        #TODO: chnage is so the batches are selected randomly
        x_batches = x.reshape(x.shape[0] // batch_nb, batch_nb, x.shape[1])
        y_batches = y.reshape(y.shape[0] // batch_nb, batch_nb, y.shape[1])

        opt = Optim(net2,self.loss_fn,self.eps)

        for epoch in range(num_epochs):
            for i in range(len(x_batches)):
                opt.step(x_batches[i],y_batches[i])
        