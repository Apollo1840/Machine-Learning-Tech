"""
see https://github.com/Apollo1840/transformer/tree/main for complete implementation.

used torch operations:
- torch.tensor
- torch.matmul
- torch.sqrt

- ().view()
- ().transpose()
- ().contiguous()

- F.softmax

- nn.Dropout()
- nn.LayerNorm()
- nn.Linear()
- nn.ReLu()

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotAttention(nn.Module):

    def __init__(self, d_k):
        super(ScaledDotAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v):
        """
        Compute the scaled dot-product attention.

        Args:
            Q (Tensor): Query tensor of shape [batch_size, num_heads, seq_len, d_k].
            K (Tensor): Key tensor of shape [batch_size, num_heads, seq_len, d_k].
            V (Tensor): Value tensor of shape [batch_size, num_heads, seq_len, d_v].
            mask (Tensor): Optional mask tensor of shape [batch_size, 1, 1, seq_len].

        Returns:
            Tuple[Tensor, Tensor]: The output tensor and the attention weights.
            output attention_weights for analysis/visualization purposes.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Multi-Head Attention Module

        NOTE:
            - in the original paper of 'attention is all you need', there is no dropout within MHA.

        Args:
            d_model (int): Dimensionality of the input and output features.
            num_heads (int): Number of attention heads, h in the 'attention is all you need paper'.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads  # this is h in the 'attention is all you need paper'.
        self.d_k = d_model // num_heads  # Dimension of each head.
        self.d_v = self.d_k  # based on the paper 'attention is all you need'.

        # Learnable projection layers for queries, keys, and values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_v * self.num_heads)  # d_model == self.d_v * self.num_heads

        # Linear layer for final output
        self.out_proj = nn.Linear(self.d_v * self.num_heads, d_model)  # d_model == self.d_v * self.num_heads

        self.attn = ScaledDotAttention(self.d_k)

        self.attention_scores = None  # Store attention weights for visualization

    def forward(self, query, key, value):
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (Tensor): Query tensor of shape [batch_size, seq_len, d_model].
            key (Tensor): Key tensor of shape [batch_size, seq_len, d_model].
            value (Tensor): Value tensor of shape [batch_size, seq_len, d_v * num_heads].

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size = query.size(0)

        # Project inputs to multi-head space
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Scaled dot-product attention
        attn_output, attention_weights = self.attn(Q, K, V)

        # Store attention weights for later visualization
        self.attention_scores = attention_weights.detach()  # Detach to avoid gradients

        # Concatenate heads and project to output space
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v * self.num_heads)
        output = self.out_proj(attn_output)

        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        Transformer Encoder Block.

        Args:
            d_model (int): Dimension of the model (embedding size).
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass of Transformer Encoder Block.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            mask (Tensor): Optional mask tensor of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Multi-Head Self-Attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x
