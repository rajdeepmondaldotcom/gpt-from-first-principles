"""Lesson 4. A neural language model: read the last few characters, score the next one."""

import math
from pathlib import Path

import torch
from torch import nn

from gpt.results import save
from gpt.tensors import cross_entropy, encode, softmax


class MLP(nn.Module):
    def __init__(self, vocab, context=8, width=24, hidden=256):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, width)  # a row of `width` numbers per character
        self.hidden = nn.Linear(context * width, hidden)
        self.out = nn.Linear(hidden, vocab)
        nn.init.zeros_(self.out.weight)  # start with no opinion: every score 0
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        """x holds character ids, shape (batch, context). Returns scores, shape (batch, vocab)."""
        rows = self.embed(x)  # (batch, context, width)
        h = torch.tanh(self.hidden(rows.flatten(1)))  # (batch, hidden)
        return self.out(h)


def batch(ids, context, size, generator):
    """Pick `size` random windows: `context` characters in, the character after them out."""
    at = torch.randint(len(ids) - context, (size,), generator=generator)
    return ids[at[:, None] + torch.arange(context)], ids[at + context]


def train(model, ids, steps, rate, size=64, generator=None):
    """Stochastic gradient descent. PyTorch's autograd now finds the slopes."""
    for _ in range(steps):
        x, y = batch(ids, model.context, size, generator)
        loss = cross_entropy(model(x), y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= rate * p.grad
    return loss.item()


@torch.no_grad()
def evaluate(model, ids):
    """Loss in nats over every window of `ids`."""
    windows = ids.unfold(0, model.context + 1, 1)  # each row: context characters, then the next
    losses = [
        cross_entropy(model(chunk[:, :-1]), chunk[:, -1]) * len(chunk)
        for chunk in windows.split(4096)
    ]
    return (sum(losses) / len(windows)).item()


@torch.no_grad()
def sample(model, start, n, generator):
    """Write n more characters. `start` is a tensor of at least `context` ids."""
    out = start.tolist()
    for _ in range(n):
        scores = model(torch.tensor([out[-model.context :]]))
        out.append(torch.multinomial(softmax(scores[0]), 1, generator=generator).item())
    return out


if __name__ == "__main__":
    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    n = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:n], ids[n:]

    def short_run(scale):
        """2,000 steps from a hidden layer whose starting weights are `scale` times larger."""
        torch.manual_seed(0)
        model = MLP(len(vocab))
        with torch.no_grad():
            model.hidden.weight *= scale
        x, _ = batch(train_ids, model.context, 256, torch.Generator().manual_seed(0))
        stuck = (model.hidden(model.embed(x).flatten(1)).tanh().abs() > 0.99).float().mean().item()
        train(model, train_ids, steps=2000, rate=0.1, generator=torch.Generator().manual_seed(0))
        return stuck, evaluate(model, val_ids)

    (stuck, short), (stuck_big, short_big) = short_run(1), short_run(20)
    print(f"normal start:    {stuck:.0%} of hidden units stuck, {short:.3f} nats after 2,000 steps")
    print(f"20x bigger start: {stuck_big:.0%} of hidden units stuck, {short_big:.3f} nats")

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = MLP(len(vocab))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"loss before training: {evaluate(model, val_ids):.3f} nats (ln 65 = {math.log(65):.3f})")
    train(model, train_ids, steps=30000, rate=0.1, generator=generator)
    train_nats, val_nats = evaluate(model, train_ids[:100000]), evaluate(model, val_ids)
    story = sample(model, encode("Once upon a time", vocab), 60, generator)
    story = "".join(vocab[i] for i in story)
    print(f"after 30,000 steps, training text: {train_nats:.3f} nats")
    print(f"unseen text:   {val_nats:.3f} nats, {val_nats / math.log(2):.3f} bits per byte")
    print(f"sample: {story!r}")
    save(
        "04_mlp_lm",
        torch=torch.__version__,
        seed=0,
        parameters=sum(p.numel() for p in model.parameters()),
        train_nats=train_nats,
        val_nats=val_nats,
        val_bits_per_byte=val_nats / math.log(2),
        sample=story,
        stuck=stuck,
        stuck_big_start=stuck_big,
        short_nats=short,
        short_nats_big_start=short_big,
    )
