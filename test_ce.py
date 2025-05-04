import numpy as np
import matplotlib.pyplot as plt
from linear import *
from non_linear import *
from sequential import *
from optim import *
from cross_entropy import *
from load_data import *

batch = 5
num_epochs = 10
X_train,Y_train,batch = get_train_data()
n = X_train.shape[0]*batch//2
#print(X_train[0].shape)
#show_digit(f"Image de : {Y_train[0]} et prediction",X_train[0])

L = CELoss()
seq = Sequential([Linear(X_train.shape[0]*batch, n),TanH(n, n),Linear(n//2, 1),Sigmoide(1, 1)])
opt = Optim(seq,L,10e-3)
opt.SGD(seq,X_train,Y_train,batch,num_epochs)

def predict(model, X):
    return model.forward(X)

prediction = predict(seq, X_train[0])
show_digit(f"Image de : {Y_train[0]} et prediction {predict}",X_train[0])
