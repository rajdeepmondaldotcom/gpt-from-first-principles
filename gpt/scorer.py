"""Lesson 28. A model of what people prefer: learn a score from pairs where one text is better."""

import torch
from torch import nn


class Scorer(nn.Module):
    """A GPT's body with one new layer on top. It reads a text and gives it a single number."""

    def __init__(self, body):
        super().__init__()
        self.body = body
        self.head = nn.Linear(body.embed.weight.size(1), 1)

    def forward(self, ids, last):
        """ids: (batch, tokens), padded. last: the place of each row's final real token."""
        x = self.body.embed(ids)
        for block in self.body.blocks:
            x = block(x)
        read = self.body.norm(x)[torch.arange(len(ids)), last]  # that place has read the whole text
        return self.head(read).squeeze(-1)


def pad(texts, fill, device="cpu"):
    """Stack lists of token ids of different lengths. Returns the grid and each row's last place."""
    ids = torch.full((len(texts), max(map(len, texts))), fill, device=device)
    for row, text in enumerate(texts):
        ids[row, : len(text)] = torch.tensor(text, device=device)
    return ids, torch.tensor([len(text) - 1 for text in texts], device=device)


def comparison_loss(better, worse):
    """Low when the better text of each pair scores higher. ln 2 when the scorer cannot tell."""
    return -nn.functional.logsigmoid(better - worse).mean()
