"""
loss function definition is very easy for pytorch, if you are very familiar with the basic torch math operations:

A interesting thing is that there is some pre-defined add-on loss in timm library, such as weight decay.

"""
import torch
from torch import nn

import timm.optim.optim_factory as optim_factory


def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss


model = nn.Module()
param_groups = optim_factory.add_weight_decay(model, 0.01)  # replace model.parameters() with param_groups
# optimizer = torch.optim.AdamW(param_groups, lr=0.01, betas=(0.9, 0.95))
# note: use AdamW instead of Adam


"""

in order to implement more custom layers, you need to master torch operations: see `viii_torch_operations.md`

"""