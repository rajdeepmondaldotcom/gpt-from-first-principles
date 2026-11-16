"""Lesson 1. Predict the next character by counting what came next before."""

import math
import random
from collections import Counter
from itertools import pairwise
from pathlib import Path

from gpt.results import save


def split(text, frac=0.9):
    """The first 90% teaches the model. The rest tests it on text it never saw."""
    n = int(len(text) * frac)
    return text[:n], text[n:]


def count(text):
    """How often each character follows each other character."""
    return Counter(pairwise(text))


def table(counts, vocab, smooth=0):
    """Turn counts into probabilities. P[a][b] is the chance that b follows a."""
    P = {}
    for a in vocab:
        row = {b: counts[a, b] + smooth for b in vocab}
        total = sum(row.values())
        P[a] = {b: c / total if total else 1 / len(vocab) for b, c in row.items()}
    return P


def loss(P, text):
    """Average surprise in nats: minus the log of the chance the model gave to what came next."""
    pairs = list(pairwise(text))
    return -sum(math.log(P[a][b]) if P[a][b] else -math.inf for a, b in pairs) / len(pairs)


def sample(P, start, n, rng):
    """Write n more characters, each drawn from the row of the one before it."""
    out = start
    for _ in range(n):
        row = P[out[-1]]
        out += rng.choices(list(row), weights=row.values())[0]
    return out


if __name__ == "__main__":
    text = Path("data/tiny_shakespeare.txt").read_text()
    train, val = split(text)
    vocab = sorted(set(text))
    P = table(count(train), vocab, smooth=1)
    bits = loss(P, val) / math.log(2)  # one character is one byte here, so bits per byte
    story = sample(P, "Once upon a time", 40, random.Random(0))
    print(f"characters: {len(vocab)}")
    print(
        f"uniform guess: {math.log(len(vocab)):.3f} nats, {math.log2(len(vocab)):.3f} bits per byte"
    )
    print(f"bigram, training text: {loss(P, train):.3f} nats")
    print(f"bigram, unseen text:   {loss(P, val):.3f} nats, {bits:.3f} bits per byte")
    print(f"sample: {story!r}")
    save("01_bigram", vocab=len(vocab), val_nats=loss(P, val), val_bits_per_byte=bits, sample=story)
