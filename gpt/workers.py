"""Lesson 20. Many workers: split each batch among processes that share their slopes."""

import time
from pathlib import Path

import torch
from torch import multiprocessing

from gpt.gpt2 import batch, loss_on
from gpt.model import GPT
from gpt.optim import AdamW
from gpt.results import save, say


def start(shape, seed, rate):
    """A model and its optimizer. The same seed gives each process the same starting weights."""
    torch.manual_seed(seed)
    model = GPT(**shape)
    return model, AdamW(model.parameters(), rate=rate)


def work(pipe, shape, seed, rate):
    """One worker. Find the slopes on its share, send them, then step with everyone's average."""
    model, optimizer = start(shape, seed, rate)
    for x, y in iter(pipe.recv, None):  # until the coordinator sends None
        model.zero_grad()
        loss_on(model, x, y).backward()
        pipe.send([p.grad for p in model.parameters()])
        for p, mean in zip(model.parameters(), pipe.recv()):
            p.grad = mean
        optimizer.step()
    pipe.send(model.state_dict())


def share(pipes, x, y):
    """One step. Split the batch, gather each worker's slopes, send back their weighted average."""
    xs, ys = x.chunk(len(pipes)), y.chunk(len(pipes))
    for pipe, part, target in zip(pipes, xs, ys):
        pipe.send((part, target))
    slopes = [pipe.recv() for pipe in pipes]
    # A loss is an average over windows, so a worker with more windows counts for more.
    mean = [
        sum(len(part) * slope for part, slope in zip(xs, same)) / len(x) for same in zip(*slopes)
    ]
    for pipe in pipes:
        pipe.send(mean)


def launch(count, shape, seed, rate):
    """Start `count` workers. Returns a pipe to each."""
    context, pipes = multiprocessing.get_context("spawn"), []
    for _ in range(count):
        ours, theirs = context.Pipe()
        context.Process(target=work, args=(theirs, shape, seed, rate), daemon=True).start()
        pipes.append(ours)
    return pipes


def finish(pipes):
    """Stop the workers. Returns the weights each one ended with."""
    for pipe in pipes:
        pipe.send(None)
    return [pipe.recv() for pipe in pipes]


if __name__ == "__main__":
    from gpt.tensors import encode

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    shape = {"vocab": len(vocab), "context": 64, "width": 128, "heads": 4, "layers": 4}
    timings = {}
    for count in (1, 2, 4):
        generator = torch.Generator().manual_seed(0)
        pipes = launch(count, shape, seed=0, rate=1e-3)
        share(
            pipes, *batch(ids, 64, 64, generator)
        )  # the first step waits for the workers to start
        began = time.perf_counter()
        for _ in range(20):
            share(pipes, *batch(ids, 64, 64, generator))
        timings[count] = (time.perf_counter() - began) / 20
        finish(pipes)
        say(f"{count} workers: {timings[count] * 1e3:.0f} ms a step on a batch of 64 windows")
    save("20_workers", torch=torch.__version__, device="cpu", seconds_per_step=timings)
