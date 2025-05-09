import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl

# Define the model again with custom weight init
class OneLayerSoftmaxNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneLayerSoftmaxNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        # Initialize with NumPy
        weight = np.random.randn(output_size, input_size).astype(np.float32)
        bias = np.random.randn(output_size).astype(np.float32)
        with torch.no_grad():
            self.fc.weight.copy_(torch.from_numpy(weight))
            self.fc.bias.copy_(torch.from_numpy(bias))

    def forward(self, x):
        return self.softmax(self.fc(x))

# Parameters
input_size = 256
output_size = 10
learning_rate = 0.01
num_epochs = 1
batch_size = 10

# same dataset
data = pkl.load(open("usps.pkl", "rb"))
x_train = np.array(data["X_train"], dtype=float)
y_train = data["Y_train"]
x_test = np.array(data["X_test"], dtype=float)
y_test = data["Y_test"]

# one-hot encoding of labels
h_y_train = np.zeros((y_train.size, 10))
h_y_train[np.arange(y_train.size), y_train] = 1

X = x_train
y = y_train

# Model, loss, optimizer
model = OneLayerSoftmaxNN(input_size, output_size)
criterion = nn.CrossEntropyLoss()  # expects raw logits, so we need to adjust model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Note: we need to remove Softmax for CrossEntropyLoss (it includes it)
class OneLayerNNRaw(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneLayerNNRaw, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        weight = np.random.randn(output_size, input_size).astype(np.float32)
        bias = np.random.randn(output_size).astype(np.float32)
        with torch.no_grad():
            self.fc.weight.copy_(torch.from_numpy(weight))
            self.fc.bias.copy_(torch.from_numpy(bias))
    def forward(self, x):
        return self.fc(x)

model = OneLayerNNRaw(input_size, output_size)

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}")
    for i in range(0, len(X), batch_size):
        inputs = X[i : i + batch_size]
        labels = y[i : i + batch_size]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Print debug info
        pred = torch.argmax(outputs, dim=1)
        print(f"Batch {i+1}")
        print(f"  Prediction: {pred.item()}, Target: {labels.item()}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients at last layer (fc.weight.grad):\n{model.fc.weight.grad}")
        print(f"  Gradients at last layer (fc.bias.grad):\n{model.fc.bias.grad}")

        # Update
        optimizer.step()
