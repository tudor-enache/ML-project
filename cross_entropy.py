"""
What does it do :
It's a different loss fuction (actually it calculates the creos entropy )
"""
import numpy as np
from projet_etu import Loss, Module

class CELoss(Loss):
    def forward(self, y, yhat):
        # y: batch x 1{s.name for s in seq}
        # yhat: batch x 1
        # output: 1
        return - y*yhat - np.log(np.sum(np.exp(yhat)))
    
    def backward(self, y, yhat):
        # output: batch x 1
        return -2 * (y - yhat)
    


