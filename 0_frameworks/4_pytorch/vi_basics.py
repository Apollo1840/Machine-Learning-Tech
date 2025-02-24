import numpy as np
import torch

from torch.nn.init import normal_
from torch.autograd import Variable
import torch.optim as optim

# ---------------------------------------------------------------------------------------
# 0. basic

# variable init
x = torch.empty(1, requires_grad=True)
print("empty x: ", x)
normal_(x)
print("random x: ", x)

# create function to optimize
y = 2 * x
print("y: ", y)

# calculate the gradient and feed it back
print("gradient of x before backward: ", x.grad)

y.backward()

print("gradient of x after backward: ", x.grad)

# optimize with backprop information
optimizer = optim.Adam([x], lr=0.1)

y = 2 * x

optimizer.zero_grad()
y.backward()

print("x before optimize: ", x)
print("y: ", 2 * x)

optimizer.step()

print("x after optimize: ", x)
print("y: ", 2 * x)

"""
Conclusion:
- torch.Tensor.backward() transfer the gradient information to the variable(weights).
- torch.nn.optim.Adam() update the variable(weights) based on transfer the gradient information.

Hence we use this train a model as following: 
"""

# ---------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------
# 2: initialization

from torch.nn.init import xavier_uniform_

# Normal: (just like before)
x = torch.empty(5, 3, requires_grad=True)
print("empty", x)

xavier_uniform_(x)
print("initialized", x)

# Direct:（if want it as variable, set requires_grad=True）
x = torch.tensor([5.5, 3])
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print("zeros", x)

x = torch.ones(5, 3, dtype=torch.long)
print("ones", x)

x = torch.rand(5, 3)
print("rand", x)

# Indirect:
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

# ----------------------------------------------------------------------------------------
# 3: operation

x = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(3, 2)

# single element: reshape
y = x.view(16)    # : reshape
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions

# get size
print(x.size(), y.size(), z.size())

# add
z = torch.add(x, x2)
print(z)

# matmul
y = torch.matmul(x3, x3.T)
# print(y.grad_fn)

# calculate gradient for each initialized variable
x3 = torch.randn(3, 2)
y = torch.matmul(x3, x3.T)

print(y.grad_fn)
y.backward()
print(x3.grad)   # grad is more like delta x