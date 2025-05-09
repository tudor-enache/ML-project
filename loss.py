import numpy as np

#--------------------------------------------------
# Base Loss Class
class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

#--------------------------------------------------
# Mean Squared Error
class MSELoss(Loss):
    def forward(self, y, yhat):
        # y: batch x 1{s.name for s in seq}
        # yhat: batch x 1
        # output: 1
        # print("Loss: ", np.mean((y - yhat) ** 2))
        return np.mean((y - yhat) ** 2)
    
    def backward(self, y, yhat):
        # output: batch x 1
        return 2 * (yhat - y)
    
#--------------------------------------------------
# Cross-Entropy Loss
class CELoss(Loss):
    def forward(self, y, yhat):
        # with one-hot encoding of labels
        return np.mean(-np.sum(y * np.log(yhat + 1e-10), axis=1))

    def backward(self, y, yhat):
        return -y / (yhat + 1e-10)