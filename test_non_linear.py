import numpy as np
import matplotlib.pyplot as plt
from linear import *
from non_linear import *

# let's work with some 2D data for nice visualization
n_samples=500
x = np.random.randn(n_samples, 2)
y = (x[:, 0] * x[:, 1] > 0).astype(int).reshape(-1, 1)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.show()

L1 = Linear(2, 4)
T = TanH(4, 4)
L2 = Linear(4, 1)
S = Sigmoide(1, 1)
L = MSELoss()

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
        y_1 = L1.forward(x_batch)
        y_t = T.forward(y_1)
        y_2 = L2.forward(y_t)
        y_hat = S.forward(y_2)

        losses.append(L.forward(y_batch, y_hat))

        L1.zero_grad()
        L2.zero_grad()

        delta_s = L.backward(y_batch, y_hat)
        delta_2 = S.backward_delta(delta_s)

        L2.backward_update_gradient(delta_2)

        delta_t = L2.backward_delta(delta_2)
        delta_1 = T.backward_delta(delta_t)

        L1.backward_update_gradient(delta_1)

        L2.update_parameters()
        L1.update_parameters()

y_hat = S.forward(L2.forward(T.forward(L1.forward(x))))

plt.scatter(np.arange(num_epochs * len(x_batches)), losses)
plt.show()