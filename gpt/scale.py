"""Lesson 19. How big should it be? Fit a law to small runs, then read off the size for a budget."""

from pathlib import Path

import torch

from gpt.cost import forward_flops
from gpt.data import read
from gpt.gpt2 import evaluate
from gpt.model import GPT
from gpt.optim import AdamW
from gpt.results import save, say
from gpt.train import train

BUDGET = 2.7e15  # multiplies and adds for the story model: about two and a half hours on this Mac


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


def steps_for(model, flops, size=32):
    """The training steps `flops` buys. A step costs about three forward passes: one, then two back."""
    return round(flops / (3 * forward_flops(model, size * model.context, model.context)))


def run(shape, flops, ids, val_ids, seed=0, size=32, peak=1e-3):
    """Train one model until it has spent `flops`. Returns its parameters, tokens and unseen loss."""
    torch.manual_seed(seed)
    model = GPT(**shape).to(ids.device)
    steps = steps_for(model, flops, size)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    generator = torch.Generator().manual_seed(seed)
    warmup = max(1, steps // 10)  # a tenth of the run, so a short run is not all warmup
    for _ in train(
        model, optimizer, ids, steps, peak=peak, generator=generator, size=size, warmup=warmup
    ):
        pass
    count = sum(p.numel() for p in model.parameters())
    return count, steps * size * model.context, evaluate(model, val_ids)


POWERS = [x / 20 for x in range(2, 31)]  # 0.10, 0.15, up to 1.50


def law(runs):
    """Fit loss = floor + a / parameters^alpha + b / tokens^beta to (parameters, tokens, loss) runs.

    For each pair of powers, least squares gives the best floor, a and b. Keep the best pair.
    """
    n, d, loss = (torch.tensor(column, dtype=torch.float64) for column in zip(*runs))
    best = None
    for alpha in POWERS:
        for beta in POWERS:
            # Parameters in millions and tokens in tens of millions keep each column near 1.
            grid = torch.stack([torch.ones_like(n), (n / 1e6) ** -alpha, (d / 1e7) ** -beta], dim=1)
            parts = torch.linalg.lstsq(grid, loss).solution
            miss = (grid @ parts - loss).pow(2).mean().sqrt().item()
            if parts.min() >= 0 and (best is None or miss < best["miss"]):
                best = {"alpha": alpha, "beta": beta, "parts": parts.tolist(), "miss": miss}
    assert best, "no pair of powers fits these runs with a floor, a and b of zero or more"
    return best


def predict(fitted, parameters, tokens):
    """The loss the law expects from a model of this size after this many tokens."""
    floor, a, b = fitted["parts"]
    return (
        floor + a * (parameters / 1e6) ** -fitted["alpha"] + b * (tokens / 1e7) ** -fitted["beta"]
    )


if __name__ == "__main__":
    cache = Path("data/cache")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ids = read(cache / "stories_train.bin").to(device)
    val_ids = read(cache / "stories_val.bin")[:200_000].to(device)
    family = [shape(32, 2), shape(48, 2), shape(64, 2), shape(96, 3), shape(128, 4), shape(192, 4)]

    def show(flops, row):
        count, tokens, loss = row
        say(f"{flops:.0e} flops, {count:>9,} parameters, {tokens:>10,} tokens: {loss:.3f} nats")
        return row

    def describe(name, fitted):
        say(
            f"{name}: floor {fitted['parts'][0]:.2f}, alpha {fitted['alpha']:.2f}, "
            f"beta {fitted['beta']:.2f}, misses its own runs by {fitted['miss']:.3f} nats"
        )

    # Each size trains at the best of three rates, found once at the middle budget.
    rates, middle = {}, []
    for one in family:
        tried = {peak: run(one, 2e13, ids, val_ids, peak=peak) for peak in (1e-3, 3e-3, 1e-2)}
        best = min(tried, key=lambda peak: tried[peak][2])
        rates[one["width"]] = best
        middle.append(tried[best])
        losses = ", ".join(f"{peak:g} gives {tried[peak][2]:.3f}" for peak in tried)
        say(f"width {one['width']}: rate {losses}")

    def measure(shapes, flops):
        return [
            show(flops, run(one, flops, ids, val_ids, peak=rates[one["width"]])) for one in shapes
        ]

    small = measure(family, 1e13) + [show(2e13, row) for row in middle] + measure(family, 4e13)
    first = law(small)
    describe("law from 18 runs", first)
    held = measure(family[2:], 8e13)  # twice the largest budget the law has seen
    for count, tokens, loss in held:
        expected = predict(first, count, tokens)
        say(
            f"held out, {count:>9,} parameters: predicted {expected:.3f}, off by {loss - expected:+.3f}"
        )
    final = law(small + held)
    describe("law from all 22 runs", final)

    choices = []
    for one in [shape(128, 4), shape(192, 4), shape(256, 6), shape(320, 8), shape(384, 8)]:
        model = GPT(**one)
        count, steps = sum(p.numel() for p in model.parameters()), steps_for(model, BUDGET)
        # A step reads train's batch: 32 windows of 256 tokens.
        expected = predict(final, count, steps * 32 * 256)
        choices.append({"shape": one, "parameters": count, "steps": steps, "predicted": expected})
        say(f"{count:>10,} parameters, {steps:>6,} steps: predicted {expected:.3f} nats")
    save(
        "19_scale",
        torch=torch.__version__,
        device=device,
        rates=rates,
        runs=small,
        held_out=held,
        first=first,
        final=final,
        budget=BUDGET,
        choices=choices,
    )
