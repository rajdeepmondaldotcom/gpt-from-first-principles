"""Lesson 19. How big should it be? Fit a law to small runs, then read off the size for a budget."""

import math
from pathlib import Path

import torch

from gpt.cost import forward_flops
from gpt.data import read
from gpt.gpt2 import evaluate
from gpt.model import GPT
from gpt.optim import AdamW
from gpt.results import save, say
from gpt.train import train


def shape(width, layers):
    """One model of the family we size: a head for each 32 of width, half as many key heads."""
    heads = max(1, width // 32)
    return {
        "vocab": 4097,
        "context": 256,
        "width": width,
        "heads": heads,
        "kv_heads": max(1, heads // 2),
        "layers": layers,
    }


def run(shape, flops, ids, val_ids, seed=0, size=32, peak=1e-3):
    """Train one model until it has spent `flops`. Returns its parameter count and unseen loss."""
    torch.manual_seed(seed)
    model = GPT(**shape).to(ids.device)
    # A training step costs about three forward passes: one forward, and twice that going back.
    steps = round(flops / (3 * forward_flops(model, size * model.context, model.context)))
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    generator = torch.Generator().manual_seed(seed)
    warmup = max(1, steps // 10)  # a tenth of the run, so a short run is not all warmup
    for _ in train(
        model, optimizer, ids, steps, peak=peak, generator=generator, size=size, warmup=warmup
    ):
        pass
    return sum(p.numel() for p in model.parameters()), steps, evaluate(model, val_ids)


def fit(xs, ys, power):
    """The curve a + b x + ... up to x to the `power` that misses the points least."""
    x = torch.tensor(xs, dtype=torch.float64)
    grid = torch.stack([x**p for p in range(power + 1)], dim=1)
    return torch.linalg.lstsq(grid, torch.tensor(ys, dtype=torch.float64)).solution.tolist()


def bottom(sizes, losses):
    """Fit a bowl to loss against ln(size). Returns the size and the loss at its lowest point."""
    a, b, c = fit([math.log(n) for n in sizes], losses, power=2)
    best = -b / (2 * c)
    return math.exp(best), a + b * best + c * best * best


if __name__ == "__main__":
    cache = Path("data/cache")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = read(cache / "stories_train.bin").to(device)
    val_ids = read(cache / "stories_val.bin")[:200_000].to(device)
    family = [shape(32, 2), shape(48, 2), shape(64, 2), shape(96, 3), shape(128, 4), shape(192, 4)]
    runs = []
    for flops in (1e13, 2e13, 4e13):
        for one in family:
            count, steps, loss = run(one, flops, ids, val_ids)
            runs.append({"flops": flops, "parameters": count, "steps": steps, "loss": loss})
            say(f"{flops:.0e} flops, {count:>9,} parameters, {steps:>5} steps: {loss:.3f} nats")
    save("19_scale", torch=torch.__version__, device=device, runs=runs)
