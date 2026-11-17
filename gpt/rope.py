"""Lesson 15. Rotary positions: turn each query and key by an angle that grows with its place."""

import math
from pathlib import Path

import torch

from gpt.results import save, say


def rope(x, pos, base=10000):
    """Pair up the entries of x and turn each pair. `pos` holds the place of each token."""
    half = x.size(-1) // 2
    speed = base ** (-torch.arange(half, device=x.device) / half)  # the first pair turns fastest
    angle = pos[:, None] * speed  # (tokens, half): place times speed
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., :half], x[..., half:]  # entry i pairs with entry i + half
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


if __name__ == "__main__":
    from gpt.bigram import split
    from gpt.experiment import compare, trial
    from gpt.gpt2 import GPT2, LayerNorm, evaluate, sample
    from gpt.model import GPT, MLP
    from gpt.optim import AdamW
    from gpt.tensors import encode
    from gpt.train import train

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    train_ids, val_ids = split(ids)
    shape = {"vocab": len(vocab), "context": 64, "width": 128, "heads": 4, "layers": 4}

    def table():
        return GPT2(**shape)

    def rotary():
        """Lesson 10's model with rotary attention and no table of places."""
        return GPT(**shape, norm=LayerNorm, mlp=MLP)

    seeds = range(5)
    a = [trial(table, train_ids, val_ids, seed, steps=600) for seed in seeds]
    b = [trial(rotary, train_ids, val_ids, seed, steps=600) for seed in seeds]
    mean, error = compare(a, b)
    say("table of places, five seeds: " + " ".join(f"{x:.4f}" for x in a))
    say("rotary, five seeds:          " + " ".join(f"{x:.4f}" for x in b))
    say(f"rotary minus table: {mean:+.4f} nats, error {error:.4f}, ratio {abs(mean) / error:.1f}")

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = rotary()
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    for _ in train(model, optimizer, train_ids, 5000, peak=3e-3, generator=generator):
        pass
    lengths = {}
    for context in (64, 128, 256):
        model.context = context  # read this many tokens at once; training used 64
        lengths[context] = evaluate(model, val_ids)
        bits = lengths[context] / math.log(2)
        say(f"reading {context} tokens: {lengths[context]:.3f} nats, {bits:.3f} bits per byte")
    model.context = 64
    story = sample(model, encode("Once upon a time", vocab), 120, generator)
    story = "".join(vocab[i] for i in story)
    say(f"sample: {story!r}")
    save(
        "15_rope",
        torch=torch.__version__,
        device="cpu",
        table=a,
        rotary=b,
        difference=mean,
        error=error,
        nats_by_length={str(k): v for k, v in lengths.items()},
        sample=story,
    )
