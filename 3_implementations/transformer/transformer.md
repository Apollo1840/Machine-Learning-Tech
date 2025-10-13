# Transformer implementation tutorial

Given:

```python

"""
- torch.tensor
- torch.matmul
- torch.sqrt

- ().view()
- ().transpose()
- ().transpose().contiguous().view()

- F.softmax

- nn.Dropout()
- nn.LayerNorm()
- nn.Linear()
- nn.ReLu()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

```

Lets implement **ScaleDotAttention** first:

```python

class ScaleDotAttention(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.d = torch.tensor(d)
    
    def forward(self, Q,  K,  V):
        """Q, K, V in shape (batch_size, num_heads, seq_len, d)"""

        D = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(self.d)
        A = F.softmax(D, dim=-1)
        return torch.matmul(A, V)

```

Based on that, lets implemente **MultiHeadAttention**:

```python

class MultiHeadAttention(nn.Module):

    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.d_k = d // n_heads

        self.sdot_att = ScaleDotAttention(self.d_k)
        self.proj_q =nn.Linear(self.d, self.d)
        self.proj_k =nn.Linear(self.d, self.d)
        self.proj_v =nn.Linear(self.d, self.d)
        self.proj_o =nn.Linear(self.d, self.d)

    def forward(self, Q, K, V):
        """Q, K, V in shape (batch_size, seq_len, d)"""
        batch_size, _, d = X.shape
        assert d == self.d

        Q = self.proj_q(Q)
        K = self.proj_k(K)
        V = self.proj_v(V)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(-3, -2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(-3, -2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(-3, -2)

        V2 = self.sdot_att(Q, K, V).transpose(-3, -2).contiguous().view(batch_size, -1, self.d)
        return self.proj_o(V2)

```

Next, lets build a **TransformerEncoderBlock**:

```python

class TransformerEncoderBlock(nn.Module):

    def __init__(self, d, n_heads, dropout_rate, d_ff):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.d_ff = d_ff

        self.mha = MultiHeadAttention(self.d, self.n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(self.d, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d)
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)
    
    def forward(self, X):
        """X in shape (batch_size, seq_len, d)"""
        X = X + self.dropout(self.mha(X, X, X))
        X = self.norm1(X)

        X = X + self.dropout(self.ffn(X))
        Y = self.norm2(X)
        return Y

```

Lets also build a **TransformerDecoderBlock**:


```python

class TransformerDecoderBlock(nn.Module):

    def __init__(self, d, n_heads, dropout_rate, d_ff):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.d_ff = d_ff

        self.mha1 = MultiHeadAttention(self.d, self.n_heads)
        self.mha2 = MultiHeadAttention(self.d, self.n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(self.d, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d)
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)
        self.norm3 = nn.LayerNorm(self.d)
    
    def forward(self, XD, XE):
        """XD, XE in shape (batch_size, seq_len, d), XD is from the decoder, XE is from the encoder"""

        XD = XD + self.dropout(self.mha1(XD, XD, XD))
        XD = self.norm1(XD)

        XD = XD + self.dropout(self.mha2(XD, XE, XE))
        XD = self.norm2(XD)

        Y = XD + self.dropout(self.ffn(XD))
        Y = self.norm3(Y)
        return Y

```

