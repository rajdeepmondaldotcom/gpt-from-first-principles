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
    assert numbers.numel() % 2 == 0, "two numbers to a byte needs an even count"
    low, high = (numbers.flatten() + 8).to(torch.uint8).view(-1, 2).unbind(dim=1)
    return low | (high << 4)


def unpack(packed, shape):
    pairs = torch.stack([packed & 15, packed >> 4], dim=1)
    return (pairs.to(torch.int8) - 8).view(shape)


class Quantized(nn.Module):
    """A linear layer whose grid is stored as small whole numbers and turned back when used."""

    def __init__(self, layer, bits):
        super().__init__()
        assert layer.bias is None, "today's layers have no bias, and this one would lose its own"
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


if __name__ == "__main__":
    import copy
    import time
    from pathlib import Path

    from gpt.data import documents, read
    from gpt.evaluate import interval, losses
    from gpt.generate import generate
    from gpt.results import save, say
    from gpt.story import load

    cache = Path("data/cache")
    model, words = load(cache / "story")
    unseen = documents(read(cache / "stories_val.bin"), words.end)[:500]
    generator = torch.Generator().manual_seed(0)
    prompt = words.encode("Once upon a time")

    def size(net):
        """The bytes a model's numbers take, with a grid that two layers share counted once."""
        return sum({t.data_ptr(): t.nbytes for t in net.state_dict().values()}.values())

    def speed(net):
        """Tokens a second, writing 200 tokens. The fastest of three tries."""
        took = []
        for _ in range(3):
            began = time.perf_counter()
            generate(net, prompt, 200, temperature=0)
            took.append(time.perf_counter() - began)
        return 200 / min(took)

    full = [losses(model, doc, 256, 256).tolist() for doc in unseen]
    nats = sum(map(sum, full)) / sum(map(len, full))
    say(
        f"32 bits: {size(model) / 1e6:.1f} MB, {nats:.3f} nats, {speed(model):,.0f} tokens a second"
    )
    numbers = {"32": {"bytes": size(model), "nats": nats}}
    for bits in (8, 4):
        small = shrink(copy.deepcopy(model), bits)
        worst = max(
            (
                (
                    layer.weight - (unpack(q.numbers, q.shape) if q.four else q.numbers) * q.scale
                ).abs()
                / q.scale
            )
            .max()
            .item()
            for block, shrunk in zip(model.blocks, small.blocks)
            for part, less in ((block.attn, shrunk.attn), (block.mlp, shrunk.mlp))
            for (_, layer), (_, q) in zip(part.named_children(), less.named_children())
            if isinstance(q, Quantized)
        )
        scored = [losses(small, doc, 256, 256).tolist() for doc in unseen]
        gaps = [[s - f for s, f in zip(new, old)] for new, old in zip(scored, full)]
        gap = sum(map(sum, gaps)) / sum(map(len, gaps))
        low, high = interval(gaps, generator)
        say(
            f"{bits} bits: {size(small) / 1e6:.1f} MB, {gap:+.4f} nats "
            f"(95% interval {low:+.4f} to {high:+.4f}), {speed(small):,.0f} tokens a second"
        )
        say(f"{bits} bits: no weight moved by more than {worst:.2f} of a step")
        numbers[str(bits)] = {"bytes": size(small), "gap": gap, "interval": [low, high]}
    save("34_quantize", torch=torch.__version__, device="cpu", numbers=numbers)
