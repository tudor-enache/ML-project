"""
What does it do :
It's a different loss fuction (actually it calculates the creos entropy )
"""
import numpy as np
from projet_etu import Loss, Module
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits)) 
    return exp_logits / np.sum(exp_logits)

class CELoss(Loss):

    
    def forward(self, y, yhat):
        # y: batch x 1{s.name for s in seq}
        # yhat: batch x 1
        # output: 1
        #return  - y*yhat + np.log(np.sum(np.exp(yhat)))
        yhat_softmax = softmax(yhat)
        return -np.sum(y * np.log(yhat_softmax + 1e-12)) 
    
    def backward(self, y, yhat):
        # output: batch x 1
        yhat_softmax = softmax(yhat)
        return yhat_softmax - y
    
    
    


