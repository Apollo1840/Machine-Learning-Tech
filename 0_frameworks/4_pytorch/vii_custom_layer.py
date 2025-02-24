"""

It is super easy to create a custom layer in pytorch.

Just like build a model, you just need build a nn.Module

"""

import math
import torch
import torch.nn as nn
import numpy as np


class MyLinearLayer(nn.Module):

    def __init__(self, size_in, size_out):
        super().__init__()

        self.size_in, self.size_out = size_in, size_out

        # require grad is default set to True
        # nn.Parameter create gradient for the tensor and register this tensor to the model.
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)

        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        nn.init.uniform_(self.bias)  # bias init

    def forward(self, x):
        w_times_x = torch.mm(x, self.weights.t())  # matrix multiplication
        y = torch.add(w_times_x, self.bias)  # w times x + b
        return y


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        self.linear = MyLinearLayer(256, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256)
        y = self.linear(x)
        return y


"""
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


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        """
         transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，
         而view操作要求tensor的内存连续存储，所以需要contiguous来返回一个contiguous copy
        """

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)

        return y
