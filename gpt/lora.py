"""Lesson 26. LoRA: leave every weight alone and train a small correction beside a few grids."""

from pathlib import Path

import torch
from torch import nn

from gpt.results import save, say


class LoRA(nn.Module):
    """A frozen linear layer, plus the product of two thin grids that are trained."""

    def __init__(self, layer, rank, scale=2.0):
        super().__init__()
        self.layer, self.scale = layer, scale
        self.down = nn.Parameter(torch.randn(rank, layer.in_features) / rank)  # width to rank
        self.up = nn.Parameter(torch.zeros(layer.out_features, rank))  # zero: no change at first

    def forward(self, x):
        return self.layer(x) + self.scale * (x @ self.down.T @ self.up.T)


GRIDS = ("q", "k", "v", "proj", "gate", "wide", "narrow")  # each linear layer in a block


def add_lora(model, rank, names=("q", "v")):
    """Freeze the whole model, then put a correction beside each grid in a block that has one of
    these names. The two named here make the queries and the values."""
    model.requires_grad_(False)
    for block in model.blocks:
        for part in (block.attn, block.mlp):
            for name, layer in list(part.named_children()):
                if name in names:
                    setattr(part, name, LoRA(layer, rank))


@torch.no_grad()
def merge(model):
    """Add each correction into its grid and put the plain layer back. The model computes the same."""
    for block in model.blocks:
        for part in (block.attn, block.mlp):
            for name, one in list(part.named_children()):
                if isinstance(one, LoRA):
                    one.layer.weight += one.scale * one.up @ one.down
                    setattr(part, name, one.layer)


if __name__ == "__main__":
    from gpt.finetune import STEPS, examples_from, measure, plain_gap, prepare, tune
    from gpt.optim import AdamW
    from gpt.story import load

    cache = Path("data/cache")
    model, words = load(cache / "story")
    generator = torch.Generator().manual_seed(0)
    pairs, held, unseen = prepare(cache, words)
    examples = examples_from(pairs, words, model.context)
    whole = sum(p.numel() for p in model.parameters())
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    before, plain_before = measure("before", model, words, held, unseen, generator)
    start, runs = generator.get_state(), {}  # each run then draws the batches lesson 25 drew
    # One change at a time from the first run: eight times the rank, then every grid.
    for name, rank, names in (
        ("rank 8, queries and values", 8, ("q", "v")),
        ("rank 64, queries and values", 64, ("q", "v")),
        ("rank 8, each grid in a block", 8, GRIDS),
    ):
        model, _ = load(cache / "story")
        generator.set_state(start)
        torch.manual_seed(0)  # each run draws its first A from the same seed
        add_lora(model, rank, names)
        trained = [p for p in model.parameters() if p.requires_grad]
        count = sum(p.numel() for p in trained)
        say(f"{name}: {count:,} of {whole:,} parameters train, {4 * count / 1e6:.2f} MB to store")
        model.to(device)  # as lesson 25 trained
        optimizer = AdamW(trained, rate=0.0, decay=0.0)
        for step, loss in tune(
            model,
            optimizer,
            examples,
            STEPS,
            peak=2e-3,
            generator=generator,
            pad=words.end,
            size=32,
        ):
            if step % 500 == 0:
                say(f"step {step}: {loss:.3f} nats on the answers")
        model.to("cpu")
        after, plain_after = measure(name, model, words, held, unseen, generator)
        gap, (low, high) = plain_gap(plain_after, plain_before, generator)
        say(
            f"{name}: plain stories, after minus before, {gap:+.3f} nats, "
            f"95% interval {low:+.3f} to {high:+.3f}"
        )
        runs[name] = {"trained": count, "after": after, "plain_gap": gap, "interval": [low, high]}

    window = unseen[0][None, :64]
    with torch.no_grad():
        apart = model(window)
        merge(model)
        together = model(window)
    say(
        f"merged against unmerged, largest difference in a score: {(apart - together).abs().max():.6f}"
    )
    save("26_lora", torch=torch.__version__, device=device, whole=whole, before=before, runs=runs)
