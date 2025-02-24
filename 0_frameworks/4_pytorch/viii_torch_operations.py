"""
This script guides through:
- Common sub-modules
- Common torch operations
    - reshape (Magic Cube)
    - joint
    - calculation (Linear algebra)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------
# Common sub-modules

fc = nn.Linear(1, 3)  # (in_features, out_features)
conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))  # (in_channels, out_channels, kernel_size, stride)
gru = nn.GRU(32, 256, bidirectional=True)  # (input_size, hidden_size)
mha = nn.MultiheadAttention(256, 8)  # (hidden_size, num_heads)

a = nn.ReLU()  # or y = F.relu(x)
pooling = nn.MaxPool2d(2, 2)  # (kernel_size, stride)
gpooling = nn.AdaptiveMaxPool2d((128, 32))  # (output_size)

flatten = nn.Flatten()

dp = nn.Dropout(0.5)
bn = nn.BatchNorm1d(32)  # (num_channels)
ln = nn.LayerNorm(32)  # (normalized_shape)


# ----------------------------------------------------------------
# Common torch operations: Magic cube

"""
Magic Cube 1

Reshape: 
torch.reshape() and torch.view() both change the shape of the tensor without changing the total number of variables.
torch.view() is faster but need variable to be contiguous while torch.reshape() does it automatically. 

reshape is always channel-last ordering
"""

# Examples:

x = torch.randn(32, 64, 7, 7)  # (batch_size, channels, height, width)

#  Reshape as flatten:
flattened = x.view(32, -1)  # (32, 64*7*7)

x = torch.randn(32, 64)  # (batch_size, channels, height, width)

# Reshape for distributed processing:
distributed_x = x.view(32, 8, 8)
distributed_y = nn.Linear(8, 8)(distributed_x)
y = distributed_y.view(32, 64)

"""
Magic Cube 2

Squeeze & Unsqueeze, add/remove dimensions where size is one.

P.S: can be implemented by review()

"""

# squeeze often used in loss
x = torch.randn(32, 1, 7, 7)
y = torch.randn(32, 7, 7)
loss = nn.MSELoss()(x.squeeze(1), y)

# unsqueeze for board-casting
img = torch.randn(64, 28, 28)  # (batch_size, height, width)
x = img.unsqueeze(1)

# NOTE: The Boardcasting is from last dimension of the second input,
# if it is matched, then go forward, it is 1, raise to match.
# example: A.shape = (64, 100, 8), B.shape = (100, 1), A + B shape is (64, 100, 8)

"""
Magic Cube 3

Transpose and Permute

Transpose used to swap the dimension, Permute is a more general form of it. 

"""

x = torch.randn(32, 100, 256)  # (batch_size, sequence_length, embedding_dim)

# LSTM expects input of shape (sequence_length, batch_size, embedding_dim)
lstm_input = x.transpose(0, 1)

x = torch.randn(32, 8, 128)  # (batch_size, heads, d_model)

# MHA often require (heads, batch_size, d_model)
x = x.permute(1, 0, 2)


"""
Example:

- We have same shape of projected query, key and value (batch_size, seq_len, d_model), where d_model = d_k * n_heads.
- We have self-attention layer which takes (.., seq_len, d_k) shape tensor as inputs and output the same.
- We want (batch_size, seq_len, d_model) as output.

How do we implement this? do not explicitly use seq_len.

"""

x = torch.randn(32, 100, 256)  # d_k = 64, n_heads = 4
attn_layer = nn.Linear(64, 64)

x = x.view(32, -1, 4, 64)   # split the d_model dimension
x = x.transpose(1, 2)       # raise n_heads dimension up
y = attn_layer(x)
y = y.transpose(1, 2)       # take n_heads dimension back, now we have (batch_size, seq_len, num_heads, d_model)
y = y.contiguous().view(32, -1, 256)   # apply contiguous after transpose, so that we can use view to reshape it.


# ----------------------------------------------------------------
# Common torch operations: Joint

# Concatenate
result = torch.cat((x, x, x), dim=0)

# repeat
# dims: A tuple of integers specifying the number of times to repeat along each dimension.
x = torch.tensor([1, 2, 3])
result = torch.tile(x, dims=(3,))  # Repeat 3 times along the only dimension

x = torch.tensor([[1, 2],
                  [3, 4]])

result = torch.tile(x, dims=(2, 3))  # Repeat 2 times along rows, 3 times along columns

# boardcasting
x = torch.tensor([1, 2, 3])  # Shape: (3,)
result = x.expand(4, 3)  # Repeat along dim 0


# ----------------------------------------------------------------
# Common torch operations: Linear algebra

"""

Torch operations:

    elementwise_add:    torch.add or +
    elementwise_mul:    torch.mul or *
    elementwise_div:    torch.div or /
    elementwise_power:  torch.pow, torch.exp
    
    max:                torch.amax
    
    matmul:             torch.mm
    einsum:             torch.einsum
                        
"""

# (not recommend) Einstein summation for 'Matrix multiplication'

# Inner(dot) product
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

result = torch.einsum('i,i -> ', a, b)  # Summation over index i
print(result)  # tensor(32)

# Outter Product
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

result = torch.einsum('i,j -> ij', a, b)  # No summation (no repeated index)

"""
- Transpose:                torch.einsum('ij -> ji', A)

- Linear transform:         torch.einsum('ij, j -> i', A, x)
- Batched Linear transform: torch.einsum('ij,bj -> bi', A, X) 

- Batched inner product:    torch.einsum('bhi,bhi -> bh', Q, K)
  (often used in attention calculation)
  
"""

# Matrix Multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

result = torch.einsum('ij,jk->ik', a, b)  # Tensor([[19, 22], [43, 50]])








