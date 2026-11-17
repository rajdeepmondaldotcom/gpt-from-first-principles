"""Lesson 3. Tensors: the same arithmetic on whole grids of numbers at once."""

import math
import time
from pathlib import Path

import torch

from gpt.results import save, say


def softmax(scores):
    """Lesson 2's softmax for a grid: each row of scores becomes a row of chances."""
    scores = scores - scores.max(dim=-1, keepdim=True).values  # same chances, no overflow
    exps = scores.exp()
    return exps / exps.sum(dim=-1, keepdim=True)


def cross_entropy(scores, targets):
    """Average surprise: minus the log of the chance each row gave to its target."""
    scores = scores - scores.max(dim=-1, keepdim=True).values
    log_chances = scores - scores.exp().sum(dim=-1, keepdim=True).log()
    return -log_chances[torch.arange(len(targets)), targets].mean()


def cross_entropy_slopes(scores, targets):
    """The slope of that loss for each score, worked out by hand."""
    slopes = softmax(scores)
    slopes[torch.arange(len(targets)), targets] -= 1
    return slopes / len(targets)


def encode(text, vocab):
    """Text to a tensor of whole numbers: each character becomes its place in the vocabulary."""
    place = {c: i for i, c in enumerate(vocab)}
    return torch.tensor([place[c] for c in text], dtype=torch.long)


def learn_bigram(ids, vocab_size, steps, rate, batch=None, generator=None):
    """Lesson 2's loop. A step handles all the pairs at once, or a random batch of them."""
    W = torch.zeros(vocab_size, vocab_size)
    pairs = len(ids) - 1
    for _ in range(steps):
        at = (
            torch.arange(pairs)
            if batch is None
            else torch.randint(pairs, (batch,), generator=generator)
        )
        x, y = ids[at], ids[at + 1]
        slopes = cross_entropy_slopes(W[x], y)  # one row of slopes per pair
        # Each pair read row x of W, so its slopes are added to that row.
        W -= rate * torch.zeros_like(W).index_add_(0, x, slopes)
    return W


def sample(W, start, n, generator):
    """Write n more characters from a table of scores W. `start` is a tensor of ids."""
    out = start.tolist()
    for _ in range(n):
        out.append(torch.multinomial(softmax(W[out[-1]]), 1, generator=generator).item())
    return out


if __name__ == "__main__":
    from gpt.bigram import split

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    train, val = split(ids)

    def seconds_per_step(pairs, repeats):
        start = time.perf_counter()
        learn_bigram(train[: pairs + 1], len(vocab), steps=repeats, rate=1.0)
        return (time.perf_counter() - start) / repeats

    small, full = seconds_per_step(2000, 100), seconds_per_step(len(train) - 1, 3)
    say(f"one step on 2,000 pairs:     {small * 1000:.1f} ms")
    say(f"one step on {len(train) - 1:,} pairs: {full:.2f} s")

    generator = torch.Generator().manual_seed(0)
    start = time.perf_counter()
    W = learn_bigram(train, len(vocab), steps=2000, rate=50.0, batch=2000, generator=generator)
    seconds = time.perf_counter() - start
    train_nats = cross_entropy(W[train[:-1]], train[1:]).item()
    val_bits = cross_entropy(W[val[:-1]], val[1:]).item() / math.log(2)
    story = sample(W, encode("Once upon a time", vocab), 40, generator)
    story = "".join(vocab[i] for i in story)
    say(f"2,000 steps on batches of 2,000 pairs: {seconds:.0f} s")
    say(f"learned bigram, training text: {train_nats:.3f} nats")
    say(f"learned bigram, unseen text:   {val_bits:.3f} bits per byte")
    say(f"sample: {story!r}")
    save(
        "03_tensors",
        torch=torch.__version__,
        step_2000_pairs_ms=small * 1000,
        step_all_pairs_s=full,
        train_nats=train_nats,
        val_bits_per_byte=val_bits,
        sample=story,
    )
