"""Lesson 17. What it costs: count the memory and the arithmetic before spending them."""

import time
from pathlib import Path

import torch

from gpt.attention import attend
from gpt.results import save, say


def forward_flops(model, tokens, context):
    """Multiplies and adds in a forward pass over `tokens` tokens, read in windows of `context`."""
    grids = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    width, layers = model.embed.weight.size(1), len(model.blocks)
    weights = 2 * grids  # each token meets each weight once: one multiply and one add
    reading = 4 * context * width * layers  # each token scores every key, then mixes every value
    return (weights + reading) * tokens


def cache_bytes(model, tokens):
    """What lesson 16's cache holds for `tokens` tokens: a key and a value in each block."""
    attn = model.blocks[0].attn
    return 2 * len(model.blocks) * attn.k.out_features * 4 * tokens  # 4 bytes a number


def layer_bytes(model, ids):
    """Run one forward pass and add up the bytes of each layer's result."""
    total = 0

    def note(layer, inputs, out):
        nonlocal total
        total += out.nbytes

    hooks = [m.register_forward_hook(note) for m in model.modules() if not list(m.children())]
    model(ids)
    for hook in hooks:
        hook.remove()
    return total


def attend_in_tiles(q, k, v, mask, tile):
    """Lesson 6's `attend`, `tile` queries at a time. The whole grid of scores never exists."""
    rows = range(0, q.size(-2), tile)
    return torch.cat(
        [attend(q[..., i : i + tile, :], k, v, mask[i : i + tile])[0] for i in rows], -2
    )


if __name__ == "__main__":
    from gpt.gpt2 import batch, loss_on
    from gpt.model import GPT
    from gpt.optim import AdamW
    from gpt.tensors import encode

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = GPT(vocab=len(vocab), context=64, width=128, heads=4, layers=4, kv_heads=2)
    optimizer = AdamW(model.parameters(), rate=1e-3)
    x, y = batch(ids, model.context, 32, generator)
    count = sum(p.numel() for p in model.parameters())

    def seconds(work, repeats=20):
        """The fastest of `repeats` timings: the one least disturbed by the rest of the machine."""
        best = float("inf")
        for _ in range(repeats):
            began = time.perf_counter()
            work()
            best = min(best, time.perf_counter() - began)
        return best

    def step():
        model.zero_grad()
        loss_on(model, x, y).backward()
        optimizer.step()

    with torch.no_grad():
        forward = seconds(lambda: model(x))
    whole = seconds(step)
    flops = forward_flops(model, x.numel(), model.context)
    say(f"parameters: {count:,}, {4 * count / 1e6:.1f} MB of weights")
    say(f"training state: weights, slopes and two averages, {16 * count / 1e6:.1f} MB")
    say(f"layer results for one batch of 32 x 64 tokens: {layer_bytes(model, x) / 1e6:.1f} MB")
    say(f"forward pass over 2,048 tokens: {flops / 1e9:.2f} billion multiplies and adds")
    say(f"forward pass: {forward * 1e3:.0f} ms, {flops / forward / 1e9:.0f} billion a second")
    say(f"training step: {whole * 1e3:.0f} ms, {whole / forward:.1f} times the forward pass")
    say(f"three hours at this speed: {3 * 3600 / whole * x.numel() / 1e6:,.0f} million tokens")
    one = forward_flops(model, 1, 64)
    moved = 4 * count + cache_bytes(model, 64)
    say(f"cache: {cache_bytes(model, 1):,} bytes a token")
    say(f"one new token, 64 cached: {one / moved:.2f} multiplies and adds for each byte read")
    say(f"one training step: {3 * flops / (16 * count):,.0f} for each byte of training state")
    save(
        "17_cost",
        torch=torch.__version__,
        device="cpu",
        parameters=count,
        forward_seconds=forward,
        step_seconds=whole,
        forward_flops=flops,
    )
