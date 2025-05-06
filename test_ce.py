import numpy as np
import matplotlib.pyplot as plt
from linear import *
from non_linear import *
from sequential import *
from optim import *
from cross_entropy import *
from load_data import *

batch = 9
num_epochs = 10
X_train,Y_train,batch = get_train_data()
Y_train = labels_to_one_hot(Y_train)
print(Y_train.shape)
n = X_train.shape[1]//2
print(X_train.shape)
#show_digit(f"Image de : {Y_train[0]} et prediction",X_train[0])

L = CELoss()
seq = Sequential([Linear(X_train.shape[1], n),TanH(n, n),Linear(n, 10),Sigmoide(10, 10)])
print("Sequential is defined")
opt = Optim(seq,L,10e-3)
print("Optim is also defined")
losses,x_batches = opt.SGD(seq,X_train,Y_train,batch,num_epochs)
print("Training is done")
def predict(model, X):
    return model.forward(X)

import matplotlib.pyplot as plt
print(len(x_batches)*num_epochs)
print(np.array(losses).shape)
#plt.scatter(np.arange(len(x_batches)*num_epochs), losses)
#plt.suptitle("Loss")

#plt.tight_layout()
#plt.show()

prediction = predict(seq, X_train[0])
show_digit(f"Image de : {Y_train[0]} et prediction {prediction}",X_train[0])

prediction = predict(seq, X_train[19])
show_digit(f"Image de : {Y_train[19]} et prediction {prediction}",X_train[19])
