import numpy as np
import torch
import torch.nn as nn
from cross_entropy import CELoss  # Assuming you have this implemented

ce_Loss = CELoss()
ce_nn = nn.CrossEntropyLoss()

# Example logits (raw model outputs)
y_pred_logits = np.array([2.0, 1.0, 0.1, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0])  # Example logits

# PyTorch expects class indices, not one-hot encoded labels
y_true_index = np.array([1])  # Class index for digit 1

# Convert to PyTorch tensors
y_pred_logits_tensor = torch.tensor(y_pred_logits, dtype=torch.float32)
y_true_tensor = torch.tensor(y_true_index, dtype=torch.int64)

# Calculate loss using PyTorch
loss = ce_nn(y_pred_logits_tensor.unsqueeze(0), y_true_tensor)  # Add batch dimension
print("Cross-Entropy Loss (PyTorch):", loss.item())

# Calculate loss using your custom implementation
y_true_one_hot = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # One-hot encoded true label for digit 1
my_loss = ce_Loss.forward(y_true_one_hot, y_pred_logits)
print("My Cross-Entropy Loss:", my_loss)

# Check gradients using the backward function
my_grad = ce_Loss.backward(y_true_one_hot, y_pred_logits)
print("My Gradient:", my_grad)

# Check gradients using PyTorch
# PyTorch's CrossEntropyLoss does not expose gradients directly, so we need to compute them manually
# First, we need to get the softmax probabilities
y_pred_softmax = torch.softmax(y_pred_logits_tensor, dim=0)
# Then compute the gradient manually
p = y_pred_softmax
grad_pytorch = p.clone()
grad_pytorch[y_true_tensor] -= 1  # Subtract 1 from the true class
print("PyTorch Gradient:", grad_pytorch.numpy())