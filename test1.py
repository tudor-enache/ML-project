import numpy as np
from step1 import *

batch = 1
input = 3
output = 2

x = np.random.randn(batch, input)
print(x)

y = np.random.rand(batch, output)
print(y)

p = Linear(input, output)
loss = MSELoss()

print(loss.forward(p.forward(x), y))