import numpy as np
import matplotlib.pyplot as plt
from linear import *
from non_linear import *

# let's work with some 2D data for nice visualization
n_samples=500
x = np.random.randn(n_samples, 2)
y = (x[:, 0] * x[:, 1] > 0).astype(int).reshape(-1, 1)


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
print(y_hat)

figure, axis = plt.subplots(1, 3)
axis[0].scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="k")
axis[0].set_title("Data")

axis[1].scatter(np.arange(num_epochs * len(x_batches)), losses, s = 5)
axis[1].set_title("Losses")


losses_array = np.array(losses)
losses_array = losses_array.reshape(num_epochs,len(x_batches))
mean_losses = np.mean(losses_array,axis = 0)

#I want for each betch the average loss
#axis[2].scatter(np.arange(num_epochs), mean_losses)
#axis[2].set_title("Mean Losses")

axis[2].scatter(np.arange(len(x_batches)), losses[-len(x_batches):])
axis[2].set_title("last Losses")

plt.tight_layout()
plt.show()