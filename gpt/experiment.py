"""Lesson 14. Running an experiment: tell a real difference from the luck of a seed."""

import math
from pathlib import Path

import torch

from gpt.gpt2 import evaluate
from gpt.optim import AdamW
from gpt.results import save, say
from gpt.train import train


def trial(make, ids, val_ids, seed, steps, peak=3e-3):
    """One run. The seed fixes the starting weights and the batches. Returns the loss on unseen text."""
    torch.manual_seed(seed)
    model = make()
    generator = torch.Generator().manual_seed(seed)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    for _ in train(model, optimizer, ids, steps, peak=peak, generator=generator):
        pass
    return evaluate(model, val_ids)


def compare(a, b):
    """Results of two settings, run on the same seeds. Returns the mean of b - a, and its error."""
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    mean = sum(diffs) / len(diffs)
    spread = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    return mean, spread / math.sqrt(len(diffs))  # the error shrinks as the seeds grow


if __name__ == "__main__":
    from gpt.gpt2 import GPT2
    from gpt.tensors import encode

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    n = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:n], ids[n:]
    steps = 600

    def losses(seeds, peak=3e-3, width=128):
        """Lesson 12's small model and loop, once per seed."""
        return [
            trial(
                lambda: GPT2(vocab=len(vocab), context=64, width=width, heads=4, layers=4),
                train_ids,
                val_ids,
                seed,
                steps,
                peak,
            )
            for seed in seeds
        ]

    def report(name, a, b):
        mean, error = compare(a, b)
        say(f"{name}: {mean:+.4f} nats, error {error:.4f}, ratio {abs(mean) / error:.1f}")
        return {"difference": mean, "error": error}

    base = losses(range(5))
    say("one setting, five seeds: " + " ".join(f"{x:.4f}" for x in base))

    sweep = {}
    for peak in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        sweep[peak] = base[:3] if peak == 3e-3 else losses(range(3), peak=peak)
        say(f"rate {peak:<6g}: " + " ".join(f"{x:.4f}" for x in sweep[peak]))
    best = [min(sweep, key=lambda peak: sweep[peak][seed]) for seed in range(3)]
    say(f"best rate for each seed: {best}")

    same = report("the same setting on five new seeds", base, losses(range(5, 10)))
    narrow = report("half the width", base, losses(range(5), width=64))
    save(
        "14_experiment",
        torch=torch.__version__,
        device="cpu",
        steps=steps,
        base=base,
        sweep={str(peak): row for peak, row in sweep.items()},
        best=best,
        same=same,
        narrow=narrow,
    )
