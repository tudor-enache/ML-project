import numpy as np
import matplotlib.pyplot as plt
from linear import *
from non_linear import *
from sequential import *
from optim import *

# let's work with some 2D data for nice visualization
n_samples=500
num_epochs = 100
x = np.random.randn(n_samples, 2)
y = (x[:, 0] * x[:, 1] > 0).astype(int).reshape(-1, 1)
batch = 5

L = MSELoss()
seq = Sequential([Linear(2, 4),TanH(4, 4),Linear(4, 1),Sigmoide(1, 1)])
opt = Optim(seq,L,10e-3)
opt.SGD(seq,x,y,batch,num_epochs)

figure, axis = plt.subplots(1, 2)
axis[0].scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="k")
axis[0].set_title("Data")

#axis[1].scatter(np.arange(num_epochs * len(x_batches)), losses, s = 5)
#axis[1].set_title("Losses")




plt.tight_layout()
plt.show()