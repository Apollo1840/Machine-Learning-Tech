"""
loss function definition is very easy for pytorch, if you are very familiar with the basic torch math operations:

A interesting thing is that there is some pre-defined add-on loss in timm library, such as weight decay.

Torch operations:

    elementwise_add:    torch.add or +
    elementwise_mul:    torch.mul or *
    elementwise_div:    torch.div or /
    power:              torch.pow, torch.exp
    matmul:             torch.mm
    max:                torch.amax
    reshape:            torch.view, torch.squeeze
    transpose:          torch.transpose(input, dim0, dim1)
                        torch.permute()

    relu:               F.relu

useful sublayer:
    nn.flatten



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