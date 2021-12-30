import numpy as np
import os
import torch

print('Active CUDA Device: GPU', torch.cuda.current_device())

print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
x = torch.rand(5, 3)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # a CUDA device object
    x = x.to(device)  # or just use strings ``.to("cuda")``
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    z = x + y  # operation on GPU
    print(z)
    print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!
