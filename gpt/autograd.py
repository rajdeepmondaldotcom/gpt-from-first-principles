"""Lesson 2. Gradients: which way to move each number so the loss goes down."""

import math
from itertools import pairwise

from gpt.results import save


class Value:
    """A number that remembers how it was made."""

    def __init__(self, data, parents=(), slopes=()):
        self.data = data
        self.grad = 0.0  # how much the final result moves when this number moves
        self.parents = parents  # the Values this one was computed from
        self.slopes = slopes  # how much this number moves when each parent moves

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, k):
        return Value(self.data**k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def backward(self):
        """The chain rule, run backwards from the result to everything it was made from."""
        order, seen, stack = [], set(), [(self, False)]
        while stack:  # list every Value after all the Values it was made from
            v, parents_listed = stack.pop()
            if parents_listed:
                order.append(v)
            elif v not in seen:
                seen.add(v)
                stack.append((v, True))
                stack.extend((parent, False) for parent in v.parents)
        self.grad = 1.0
        for v in reversed(order):
            for parent, slope in zip(v.parents, v.slopes):
                parent.grad += slope * v.grad


def softmax(scores):
    """Turn any list of scores into chances: all positive, summing to 1."""
    exps = [s.exp() for s in scores]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]


def bigram_loss(W, pairs):
    """The lesson 1 loss, for a table of adjustable scores W instead of counts."""
    total = sum((-softmax(W[a])[b].log() for a, b in pairs), Value(0.0))
    return total / len(pairs)


def learn_bigram(text, steps=300, rate=5.0):
    """Start with every score at 0 and walk downhill. Returns the loss after each step."""
    vocab = sorted(set(text))
    pairs = [(vocab.index(a), vocab.index(b)) for a, b in pairwise(text)]
    W = [[Value(0.0) for _ in vocab] for _ in vocab]
    losses = []
    for _ in range(steps):
        loss = bigram_loss(W, pairs)
        loss.backward()
        for row in W:
            for w in row:
                w.data -= rate * w.grad
                w.grad = 0.0
        losses.append(loss.data)
    return losses


if __name__ == "__main__":
    import time
    from pathlib import Path

    from gpt.bigram import count, loss, table

    sentence = "the cat sat on the mat"
    counted = loss(table(count(sentence), sorted(set(sentence))), sentence)
    losses = learn_bigram(sentence)
    print(f"counted table:      {counted:.4f} nats")
    for step in (0, 1, 10, 100, 299):
        print(f"learned, step {step:>3}:  {losses[step]:.4f} nats")

    # How long would one step on all of Shakespeare take? Time 2,000 pairs and scale up.
    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    pairs = [(vocab.index(a), vocab.index(b)) for a, b in pairwise(text[:2001])]
    W = [[Value(0.0) for _ in vocab] for _ in vocab]
    start = time.perf_counter()
    bigram_loss(W, pairs).backward()
    seconds = time.perf_counter() - start
    minutes = seconds / 2000 * (len(text) - 1) / 60
    print(f"one step on 2,000 pairs of Shakespeare: {seconds:.2f} s")
    print(f"one step on the whole file would take:  {minutes:.1f} minutes")
    save("02_autograd", counted=counted, learned=losses[-1], step_2000_pairs_s=seconds)
