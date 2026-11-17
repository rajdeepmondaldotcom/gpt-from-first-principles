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


def add_lora(model, rank, names=("q", "v")):
    """Freeze the whole model, then put a correction beside the named grids of each attention."""
    model.requires_grad_(False)
    for block in model.blocks:
        for name in names:
            setattr(block.attn, name, LoRA(getattr(block.attn, name), rank))


@torch.no_grad()
def merge(model):
    """Add each correction into its grid and put the plain layer back. The model computes the same."""
    for block in model.blocks:
        for name, part in list(block.attn.named_children()):
            if isinstance(part, LoRA):
                part.layer.weight += part.scale * part.up @ part.down
                setattr(block.attn, name, part.layer)


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

    before, plain_before = measure("before", model, words, held, unseen, generator)
    add_lora(model, rank=8)
    trained = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in trained)
    say(f"trained: {count:,} of {whole:,} parameters, {4 * count / 1e6:.2f} MB to store")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)  # as lesson 25 trained
    optimizer = AdamW(trained, rate=0.0, decay=0.0)
    for step, loss in tune(
        model, optimizer, examples, STEPS, peak=2e-3, generator=generator, pad=words.end, size=32
    ):
        if step % 250 == 0:
            say(f"step {step}: {loss:.3f} nats on the answers")
    model.to("cpu")
    after, plain_after = measure("after", model, words, held, unseen, generator)
    gap, (low, high) = plain_gap(plain_after, plain_before, generator)
    say(
        f"plain stories, after minus before: {gap:+.3f} nats, 95% interval {low:+.3f} to {high:+.3f}"
    )

    window = unseen[0][None, :64]
    with torch.no_grad():
        apart = model(window)
        merge(model)
        together = model(window)
    say(
        f"merged against unmerged, largest difference in a score: {(apart - together).abs().max():.6f}"
    )
    save(
        "26_lora",
        torch=torch.__version__,
        device=device,
        trained=count,
        whole=whole,
        before=before,
        after=after,
        plain_gap=gap,
        plain_interval=[low, high],
    )
