import numpy as np
import torch

from torch.nn.init import normal_
from torch.autograd import Variable
import torch.optim as optim

# 0. basic
# variable init
x = torch.empty(1, requires_grad=True)
normal_(x)
print(x)

# create function to optimize
y = 2 * x
print(y)

# calculate the gradient and feed it back
print(x.grad)
y.backward()
print(x.grad)

# optimize with backprop information
y = 2 * x


# 1. basic train
# data:
datagen = zip([torch.rand(1, 4) for _ in range(12)], [torch.zeros(1) for _ in range(12)])

# variable(weights)
W = Variable(torch.randn(4, 1), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)
model = lambda x: torch.matmul(x, W) + b

optimizer = optim.Adam([W, b])

for x, y in datagen:
    optimizer.zero_grad()

    pred = model(x)
    loss = (pred - y) ** 2

    loss.backward()
    optimizer.step()

    print(loss)

# 2: initialization
# if want variable, set requires_grad=True
x = torch.empty(5, 3)
print("empty", x)

x = torch.rand(5, 3)
print("rand", x)

x = torch.zeros(5, 3, dtype=torch.long)
print("zeros", x)

x = torch.tensor([5.5, 3])
print(x)

# from numpy
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
print(b.numpy())

# from torch tensor
x = torch.randn_like(b, dtype=torch.float)  # override dtype!
print(x)

x = x.new_ones(5, 3)      # new_* methods take in sizes
print(x)


# 3: operation
# single element: reshape
x = torch.randn(4, 4)
y = x.view(16)  # : reshape
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# add
x = torch.rand(5, 3)
y = torch.rand(5, 3)
z = torch.add(x, y)
print(z)

# matmul
x = torch.ones(1, 2, requires_grad=True)
y = torch.matmul(x, x.T)
# print(y.grad_fn)

# calculate gradient for each initialized variable
print(x.grad)
y.backward()
print(x.grad)  # grad is more like delta x
