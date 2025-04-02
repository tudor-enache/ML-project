import numpy as np
from linear import *

# define a linear dataset with 1 feature
# add some noise to the output
# try to learn it with batch-GD

# choose a number of points uniformly in the range [0, 50]
a = 3
b = 10
x = np.random.uniform(0, 20, (1000000, 1))
y = a * x + b + np.random.normal(size=x.shape)
print(x.shape)
print(y.shape)

batch = 10
x_batches = x.reshape(x.shape[0] // batch, batch, 1)
y_batches = y.reshape(y.shape[0] // batch, batch, 1)

M = Linear(1, 1)
L = MSELoss()

losses = []
weights = []
biases = []
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
    weights.append(M._parameters["w"][0][0])
    biases.append(M._parameters["b"][0][0])

y_hat = M.forward(x)
print(M)

import matplotlib.pyplot as plt
plt.scatter(x, y, color="r", label="True data")
plt.plot(x, y_hat, color="b", label="Predicted data")
plt.legend()
plt.title("Regression")
plt.show()

plt.scatter(np.arange(len(x_batches)), weights)
plt.title("Weights")
plt.show()

plt.scatter(np.arange(len(x_batches)), biases)
plt.title("Biases")
plt.show()

plt.scatter(np.arange(len(x_batches)), losses)
plt.title("Loss")
plt.show()

# Problems noticed:
# 1. If we keep increasing the range of the training data the weights don't adapt fast enough
# 2. Bias is not learned at the same rate as the weights
# => this is probably a problem with our optimizer