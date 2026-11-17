"""Lesson 25. Fine-tuning: teach a trained model to answer a prompt, and to stop."""

import torch

from gpt.tensors import cross_entropy
from gpt.train import clip, rate_at


def example(words, ask, answer):
    """One training example: the prompt, the answer, the end token. Returns ids and the answer's start."""
    prompt = words.encode(ask)
    return [*prompt, *words.encode(answer), words.end], len(prompt)


def collate(examples, pad, device="cpu"):
    """Stack examples of different lengths. `scored` marks the places whose target is answer."""
    longest = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), longest - 1), pad, device=device)
    y, scored = x.clone(), torch.zeros_like(x, dtype=torch.bool)
    for row, (ids, start) in enumerate(examples):
        ids = torch.tensor(ids, device=device)
        x[row, : len(ids) - 1], y[row, : len(ids) - 1] = ids[:-1], ids[1:]
        scored[row, start - 1 : len(ids) - 1] = (
            True  # place start - 1 predicts the answer's first token
        )
    return x, y, scored


def answer_loss(model, x, y, scored):
    """Lesson 10's loss, counted only where the target is part of an answer."""
    return cross_entropy(model(x)[scored], y[scored])


def tune(model, optimizer, examples, steps, *, peak, generator, pad, size=16, warmup=20):
    """Lesson 12's loop on batches of examples. Yields (step, loss)."""
    device = model.embed.weight.device
    for step in range(steps):
        picks = torch.randint(len(examples), (size,), generator=generator).tolist()
        optimizer.rate = rate_at(step, peak, warmup, steps)
        loss = answer_loss(model, *collate([examples[i] for i in picks], pad, device))
        model.zero_grad()
        loss.backward()
        clip(model.parameters(), 1.0)
        optimizer.step()
        yield step + 1, loss.item()
