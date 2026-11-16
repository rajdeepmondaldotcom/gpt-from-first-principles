"""Lessons 6 and 7. Attention: each token chooses what to read."""

import math

import torch
from torch import nn

from gpt.tensors import softmax


def attend(q, k, v, mask=None):
    """Each query scores each key, the scores become weights, and the weights mix the values."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, -math.inf)  # a weight of 0 wherever mask is False
    weights = softmax(scores)
    return weights @ v, weights


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)  # query, key and value for each token, in one layer
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        """x has shape (batch, tokens, width). So does the result."""
        B, T, C = x.shape
        q, k, v = (
            part.view(B, T, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, T, C/heads)
            for part in self.qkv(x).chunk(3, dim=-1)
        )
        past = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()  # row i: tokens 0 to i
        out, _ = attend(q, k, v, past)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))
