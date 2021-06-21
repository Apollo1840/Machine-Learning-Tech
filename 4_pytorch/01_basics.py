import numpy as np
import torch

from torch.autograd import Variable
import torch.optim as optim

# 0. basic
x = torch.empty(1, requires_grad=True)
y = 2 * x

y.backward()
print(x.grad)

# 1. basic train
datagen = zip([torch.rand(1, 4) for _ in range(12)], [torch.zeros(1) for _ in range(12)])

W = Variable(torch.randn(4, 1), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)

optimizer = optim.Adam([W, b])

for x, y in datagen:
    optimizer.zero_grad()

    pred = torch.matmul(x, W) + b
    loss = (pred - y) ** 2

    loss.backward()
    optimizer.step()

    print(loss)

# 2: initialization
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
# one element
x = torch.randn(4, 4)
y = x.view(16)  # : reshape
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.rand(5, 3)
y = torch.rand(5, 3)
z = torch.add(x, y)
print(z)

# get gradient
x = torch.ones(1, 2, requires_grad=True)
y = torch.matmul(x, x.T)
print(y.grad_fn)

print(x.grad)
y.backward()
print(x.grad)  # grad is more like delta x
