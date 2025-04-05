import numpy as np
import matplotlib.pyplot as plt
from linear import *
from non_linear import *
from sequential import *

# let's work with some 2D data for nice visualization
n_samples=500
x = np.random.randn(n_samples, 2)
y = (x[:, 0] * x[:, 1] > 0).astype(int).reshape(-1, 1)

L = MSELoss()
seq = Sequential([Linear(2, 4),TanH(4, 4),Linear(4, 1),Sigmoide(1, 1)])

batch = 5
x_batches = x.reshape(x.shape[0] // batch, batch, x.shape[1])
y_batches = y.reshape(y.shape[0] // batch, batch, y.shape[1])

num_epochs = 100
losses = []

for epoch in range(num_epochs):
    for i in range(len(x_batches)):
        x_batch = x_batches[i]
        y_batch = y_batches[i]

        # calculate loss at this step
        y_hat = seq.forward(x_batch)

        losses.append(L.forward(y_batch, y_hat))

        seq.zero_grad()

        delta = L.backward(y_batch, y_hat)
        seq.backward(delta)


        seq.update_parameters()


figure, axis = plt.subplots(1, 2)
axis[0].scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="k")
axis[0].set_title("Data")

axis[1].scatter(np.arange(num_epochs * len(x_batches)), losses, s = 5)
axis[1].set_title("Losses")




plt.tight_layout()
plt.show()