"""Lesson 26. LoRA: leave every weight alone and train a small correction beside a few grids."""

import torch
from torch import nn


class LoRA(nn.Module):
    """A frozen linear layer, plus the product of two thin grids that are trained."""

    def __init__(self, layer, rank, scale=2.0):
        super().__init__()
        self.layer, self.scale = layer, scale
        self.down = nn.Parameter(torch.randn(rank, layer.in_features) / rank)  # width to rank
        self.up = nn.Parameter(torch.zeros(layer.out_features, rank))  # zero: no change at first

    def forward(self, x):
        return self.layer(x) + self.scale * (x @ self.down.T @ self.up.T)


def add_lora(model, rank, names=("q", "v")):
    """Freeze the whole model, then put a correction beside the named grids of each attention."""
    model.requires_grad_(False)
    for block in model.blocks:
        for name in names:
            setattr(block.attn, name, LoRA(getattr(block.attn, name), rank))


@torch.no_grad()
def merge(model):
    """Add each correction into its grid and put the plain layer back. The model computes the same."""
    for block in model.blocks:
        for name, part in list(block.attn.named_children()):
            if isinstance(part, LoRA):
                part.layer.weight += part.scale * part.up @ part.down
                setattr(block.attn, name, part.layer)
