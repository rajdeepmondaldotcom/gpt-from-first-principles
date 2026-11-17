"""Lesson 11. Optimizers: better steps from the same slopes."""

import math
from pathlib import Path

import torch

from gpt.results import save, say


class SGD:
    """Gradient descent. With momentum, each step also keeps most of the step before it."""

    def __init__(self, params, rate, momentum=0.0):
        self.params, self.rate, self.momentum = list(params), rate, momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        for p, velocity in zip(self.params, self.velocity):
            velocity.mul_(self.momentum).add_(p.grad)
            p -= self.rate * velocity


class AdamW:
    """Adam with weight decay: a step size of its own for each parameter."""

    def __init__(self, params, rate, betas=(0.9, 0.999), decay=0.01, eps=1e-8):
        self.params, self.rate, self.betas = list(params), rate, betas
        self.decay, self.eps, self.steps = decay, eps, 0
        self.mean = [torch.zeros_like(p) for p in self.params]  # running average of the slope
        self.square = [torch.zeros_like(p) for p in self.params]  # ... and of its square

    @torch.no_grad()
    def step(self):
        self.steps += 1
        b1, b2 = self.betas
        # Both averages start at 0 and so begin too small. These two factors correct that.
        fix_mean, fix_square = 1 - b1**self.steps, 1 - b2**self.steps
        for p, mean, square in zip(self.params, self.mean, self.square):
            if p.dim() > 1:
                p *= 1 - self.rate * self.decay  # pull weight grids toward 0; leave the rest
            mean.lerp_(p.grad, 1 - b1)  # move the average a tenth of the way to the new slope
            square.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            size = square.sqrt() / math.sqrt(fix_square) + self.eps  # the slope's usual size
            p.addcdiv_(mean, size, value=-self.rate / fix_mean)  # p -= rate * mean / size


if __name__ == "__main__":
    from gpt.gpt2 import GPT2, batch, evaluate, loss_on, sample
    from gpt.tensors import encode

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    n = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:n], ids[n:]

    def run(make):
        """Lesson 10's model and data, 3,000 steps, with the optimizer that `make` builds."""
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(0)
        model = GPT2(vocab=len(vocab), context=64, width=128, heads=4, layers=4)
        optimizer = make(model.parameters())
        for _ in range(3000):
            loss = loss_on(model, *batch(train_ids, model.context, 32, generator))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        story = sample(model, encode("Once upon a time", vocab), 80, generator)
        return evaluate(model, val_ids) / math.log(2), "".join(vocab[i] for i in story)

    results = {}
    for name, make in [
        ("gradient descent", lambda p: SGD(p, rate=0.1)),
        ("with momentum", lambda p: SGD(p, rate=0.1, momentum=0.9)),
        ("AdamW", lambda p: AdamW(p, rate=1e-3)),
    ]:
        bits, story = run(make)
        results[name] = {"val_bits_per_byte": bits, "sample": story}
        say(f"{name}: {bits:.3f} bits per byte")
        say(f"  {story!r}")
    save(
        "11_optim",
        torch=torch.__version__,
        device="cpu",
        seed=0,
        steps=3000,
        runs=results,
        sample=results["AdamW"]["sample"],
    )
