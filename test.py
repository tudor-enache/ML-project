import module
import loss
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

#--------------------------------------------------
# Test 1
# Linear Regression in 2D

def test1(n_samples = 100, noise = 0.01, batch_size = 10, gradient_step = 0.01, n_epochs = 1000, verbose = False):
    # create training data
    # add Gaussian Noise to a linear dataset
    x = np.random.rand(n_samples, 1)
    w = np.random.rand(1, 1)
    y = x @ w.T + noise * np.random.randn(n_samples, 1)

    # create model
    model = module.Linear(1, 1)
    loss_func = loss.MSELoss()

    losses = []

    for epoch in range(n_epochs):
        # shuffle dataset at the start of each epoch
        order = np.random.permutation(np.arange(x.shape[0]))
        x = x[order]
        y = y[order]

        for i in range(0, x.shape[0], batch_size):
            model.zero_grad()

            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            y_pred = model.forward(x_batch)
            loss_val = loss_func.forward(y_batch, y_pred)
            losses.append(loss_val)

            delta = loss_func.backward(y_batch, y_pred)
            model.backward_update_gradient(delta)
            model.update_parameters(gradient_step)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]}")

    # look at loss function and the final prediction we learn
    plt.plot(losses)
    plt.title("Loss")
    plt.grid(True)
    plt.show()

    x_plot = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_pred = model.forward(x_plot)

    plt.scatter(x, y, color="blue", label="True Data")
    plt.plot(x_plot, y_pred, color = "red", label = "Model Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()

# test1()

#--------------------------------------------------
# Test 2
# Linear Regression in 3D

def test2(n_samples = 100, noise = 0.01, batch_size = 10, gradient_step = 0.01, n_epochs = 1000, verbose = False):
    # create training data
    # add Gaussian Noise to a linear dataset
    x = np.random.rand(n_samples, 2)
    w = np.random.rand(1, 2)
    y = x @ w.T + noise * np.random.randn(n_samples, 1)

    # create model
    model = module.Linear(2, 1)
    loss_func = loss.MSELoss()

    losses = []

    for epoch in range(n_epochs):
        # shuffle dataset at the start of each epoch
        order = np.random.permutation(np.arange(x.shape[0]))
        x = x[order]
        y = y[order]

        for i in range(0, n_samples, batch_size):
            model.zero_grad()

            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            y_pred = model.forward(x_batch)
            loss_val = loss_func.forward(y_batch, y_pred)
            losses.append(loss_val)

            delta = loss_func.backward(y_batch, y_pred)
            model.backward_update_gradient(delta)
            model.update_parameters(gradient_step)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]}")

    # look at loss function and the final prediction we learn
    plt.plot(losses)
    plt.title("Loss")
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Affichage des points de données
    ax.scatter(x[:, 0], x[:, 1], y.flatten(), color='blue', label='True Data')

    # Grille pour afficher le plan prédit
    x1_range = np.linspace(x[:, 0].min(), x[:, 0].max(), 30)
    x2_range = np.linspace(x[:, 1].min(), x[:, 1].max(), 30)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    y_pred = model.forward(X_grid).reshape(x1_grid.shape)

    # Affichage du plan
    ax.plot_surface(x1_grid, x2_grid, y_pred, color='red', alpha=0.5, label='Model Prediction')

    ax.set_title("Linear Regression (2D input)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    plt.show()

# test2()

#--------------------------------------------------
# Test 3
# Classification for XOR dataset
# Build dataset by constructing multivariate Gaussians around points (-1, -1), (-1, 1), (1, 1) and (1, -1)

def test3(n_samples = 250, sigma = 0.1, noise = 0.02, batch_size = 10, gradient_step = 0.001, n_epochs = 1000, verbose = True):
    # create training data
    # order quadrants clock-wise starting from top-left
    # concatenate and shuffle
    # use 0-1 labels
    x2 = np.random.multivariate_normal(mean = [1, 1], cov = sigma * np.eye(2), size = n_samples)
    x4 = np.random.multivariate_normal(mean = [-1, -1], cov = sigma * np.eye(2), size = n_samples)
    x1 = np.random.multivariate_normal(mean = [-1, 1], cov = sigma * np.eye(2), size = n_samples)
    x3 = np.random.multivariate_normal(mean = [1, -1], cov = sigma * np.eye(2), size = n_samples)
    x = np.vstack([x2, x4, x1, x3])
    y = np.vstack([np.ones((2 * n_samples, 1)), np.zeros((2 * n_samples, 1))])

    # add some noise
    x[:, 0] += np.random.normal(0, noise, 4 * n_samples)
    x[:, 1] += np.random.normal(0, noise, 4 * n_samples)
    
    l1 = module.Linear(2, 8)
    l2 = module.TanH()
    l3 = module.Linear(8, 1)
    l4 = module.Sigmoide()
    loss_func = loss.MSELoss()

    losses = []
    
    for epoch in range(n_epochs):
        # shuffle dataset at the start of each epoch
        order = np.random.permutation(np.arange(x.shape[0]))
        x = x[order]
        y = y[order]

        for i in range(0, 4 * n_samples, batch_size):
            l1.zero_grad()
            l3.zero_grad()

            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            y1 = l1.forward(x_batch)
            y2 = l2.forward(y1)
            y3 = l3.forward(y2)
            y4 = l4.forward(y3)
            loss_val = loss_func.forward(y_batch, y4)
            losses.append(loss_val)

            delta4 = loss_func.backward(y_batch, y4)
            delta3 = l4.backward_delta(delta4)
            delta2 = l3.backward_delta(delta3)
            delta1 = l2.backward_delta(delta2)

            l3.backward_update_gradient(delta3)
            l1.backward_update_gradient(delta1)
            l3.update_parameters(gradient_step)
            l1.update_parameters(gradient_step)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]}")

    # plot loss and decision boundary
    plt.plot(losses)
    plt.title("Loss")
    plt.show()

    cmap = plt.get_cmap("RdBu")
    xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
    ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    data = np.c_[xx.ravel(), yy.ravel()]
    y1 = l1.forward(data)
    y2 = l2.forward(y1)
    y3 = l3.forward(y2)
    y4 = l4.forward(y3)
    z = y4.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5, levels=5)

    y1 = l1.forward(x)
    y2 = l2.forward(y1)
    y3 = l3.forward(y2)
    y4 = l4.forward(y3)
    ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap=cmap, lw=0)

    plt.title("Test 3")
    plt.grid(True)
    plt.show()

# test3(batch_size=50, gradient_step=0.01)

#--------------------------------------------------
# Test 4
# Same dataset (+ uniformly distributed one) as before, but using the Sequential and Optim classes
def test4(n_samples = 250, sigma = 0.1, noise = 0.02, batch_size = 10, gradient_step = 0.001, n_epochs = 1000, verbose = True):
    # # normally distributed data
    # x2 = np.random.multivariate_normal(mean = [1, 1], cov = sigma * np.eye(2), size = n_samples)
    # x4 = np.random.multivariate_normal(mean = [-1, -1], cov = sigma * np.eye(2), size = n_samples)
    # x1 = np.random.multivariate_normal(mean = [-1, 1], cov = sigma * np.eye(2), size = n_samples)
    # x3 = np.random.multivariate_normal(mean = [1, -1], cov = sigma * np.eye(2), size = n_samples)
    # # add some Gaussian noise
    # for x in [x2, x4, x1, x3]:
    #     x[:, 0] += np.random.normal(0, noise, n_samples)
    #     x[:, 1] += np.random.normal(0, noise, n_samples)

    # uniformly distributed data in the 4 quadrants
    x2 = np.array([1, 1]) + np.random.uniform(-1 + noise, 1 - noise, (n_samples, 2))
    x4 = np.array([-1, -1]) + np.random.uniform(-1 + noise, 1 - noise, (n_samples, 2))
    x1 = np.array([-1, 1]) + np.random.uniform(-1 + noise, 1 - noise, (n_samples, 2))
    x3 = np.array([1, -1]) + np.random.uniform(-1 + noise, 1 - noise, (n_samples, 2))
    x = np.vstack([x2, x4, x1, x3])

    x = np.vstack([x2, x4, x1, x3])
    y = np.vstack([np.ones((2 * n_samples, 1)), np.zeros((2 * n_samples, 1))])
    
    model = module.Sequential([
        module.Linear(2, 8),
        module.TanH(),
        module.Linear(8, 1),
        module.Sigmoide()])
    loss_func = loss.MSELoss()
    optim = module.Optim(model, loss_func, gradient_step)

    losses = optim.SGD(x, y, batch_size, n_epochs)

    # plot loss and decision boundary
    plt.plot(losses)
    plt.title("Loss")
    plt.show()

    cmap = plt.get_cmap("RdBu")
    xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
    ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    grid = np.c_[xx.ravel(), yy.ravel()]
    y_grid = model.forward(grid)
    z = y_grid.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5, levels=5)

    ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap=cmap, lw=0)

    plt.grid(True)
    plt.title("Test 4: Sequential")
    plt.show()

# test4(noise=0.3, batch_size=50, gradient_step=0.01, n_epochs=5000)

#--------------------------------------------------
# Test 5
# find a simple test just to check the correctness of softmax implementation (not sure about backward)

def test5(batch_size = 10, gradient_step = 0.01, n_epochs = 1, verbose = True):
    data = pkl.load(open("usps.pkl", "rb"))
    x_train = np.array(data["X_train"], dtype=float)
    y_train = data["Y_train"]
    x_test = np.array(data["X_test"], dtype=float)
    y_test = data["Y_test"]

    # one-hot encoding of labels
    h_y_train = np.zeros((y_train.size, 10))
    h_y_train[np.arange(y_train.size), y_train] = 1

    np.random.seed(0)
    model = module.Sequential([
        module.Linear(256, 10),
        module.Softmax()
    ])
    loss_func = loss.CELoss()
    optim = module.Optim(model, loss_func, gradient_step)

    losses = optim.SGD(x_train, h_y_train, batch_size, n_epochs)

    # plot loss function
    plt.scatter(np.arange(len(losses)), losses)
    plt.title("Loss")
    plt.show()

    # calculate accuracy in test
    y_pred = model.forward(x_test).argmax(axis=1)
    print("Test accuracy: ", np.where(y_pred == y_test)[0].size / y_pred.size)

test5()