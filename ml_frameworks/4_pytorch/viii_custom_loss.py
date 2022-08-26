"""
loss function definition is very easy for pytorch, if you are very familiar with the basic torch math operations:


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


def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss
