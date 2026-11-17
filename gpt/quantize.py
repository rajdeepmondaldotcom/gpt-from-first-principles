"""Lesson 34. Quantization: store each weight in 8 bits, or in 4, and keep the error bounded."""

import torch
from torch import nn


def quantize(grid, bits):
    """Each row becomes small whole numbers and one scale. Returns (numbers, scales)."""
    top = 2 ** (bits - 1) - 1  # 127 for 8 bits, 7 for 4
    scale = (
        grid.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / top
    )  # the row's largest maps to top
    return (grid / scale).round().to(torch.int8), scale


def pack(numbers):
    """Two 4-bit numbers in each byte. Halves the bytes of numbers between -8 and 7."""
    low, high = (numbers.flatten() + 8).to(torch.uint8).view(-1, 2).unbind(dim=1)
    return low | (high << 4)


def unpack(packed, shape):
    pairs = torch.stack([packed & 15, packed >> 4], dim=1)
    return (pairs.to(torch.int8) - 8).view(shape)


class Quantized(nn.Module):
    """A linear layer whose grid is stored as small whole numbers and turned back when used."""

    def __init__(self, layer, bits):
        super().__init__()
        numbers, scale = quantize(layer.weight.detach(), bits)
        self.shape, self.four = numbers.shape, bits == 4
        self.register_buffer("numbers", pack(numbers) if self.four else numbers)
        self.register_buffer("scale", scale)

    def forward(self, x):
        numbers = unpack(self.numbers, self.shape) if self.four else self.numbers
        return x @ (numbers * self.scale).T


def shrink(model, bits):
    """Replace each linear layer inside the blocks with its quantized copy."""
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name, layer in list(parent.named_children()):
                if isinstance(layer, nn.Linear):
                    setattr(parent, name, Quantized(layer, bits))
    return model
