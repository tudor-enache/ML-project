import numpy as np
from linear import *

# define a linear dataset with 1 feature
# add some noise to the output
# try to learn it with batch-GD

a = 2
x = np.arange(1, 101)
y = a * x + np.random.normal(size=x.shape)
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

batch = 5
x_batches = x.reshape(x.shape[0] // batch, batch, 1)
y_batches = y.reshape(y.shape[0] // batch, batch, 1)

M = Linear(1, 1)
L = MSELoss()

losses = []
for i in range(len(x_batches)):
    x_batch = x_batches[i]
    y_batch = y_batches[i]

    # calculate loss at this step
    y_hat_batch = M.forward(x_batch)
    losses.append(L.forward(y_batch, y_hat_batch))

    M.zero_grad()
    delta = L.backward(y_batch, y_hat_batch)
    M.backward_update_gradient(delta)
    M.update_parameters()

y_hat = M.forward(x)
print(M)

import matplotlib.pyplot as plt
plt.plot(x, y, color="r", label="True data")
plt.plot(x, y_hat, color="b", label="Predicted data")
plt.legend()
plt.show()