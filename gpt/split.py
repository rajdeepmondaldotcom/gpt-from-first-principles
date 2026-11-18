"""Lesson 43. A model no one card holds: share out a layer's hidden units, and the optimizer's
averages, among workers, and check that every sum comes out as it did in one piece."""

import torch

from gpt.optim import AdamW


def shares(count, workers):
    """Cut `count` things into `workers` runs of nearly equal length: who holds which."""
    return [slice(count * i // workers, count * (i + 1) // workers) for i in range(workers)]


def deal(mlp, workers):
    """What each worker holds of lesson 16's MLP: a share of its hidden units, as their rows of
    the two grids that widen and their columns of the grid that narrows."""
    units = shares(mlp.narrow.in_features, workers)
    return [(mlp.gate.weight[u], mlp.wide.weight[u], mlp.narrow.weight[:, u]) for u in units]


def part(held, x):
    """One worker's part of the MLP's result: the whole input, through the weights it holds.
    The parts of all the workers add up to the result, and that sum is the one exchange."""
    gate, wide, narrow = held
    opened = x @ gate.T
    return (opened * torch.sigmoid(opened) * (x @ wide.T)) @ narrow.T


def dealt(params, workers, **how):
    """AdamW's two averages, shared out: worker i keeps them for each `workers`-th tensor from
    tensor i on, and steps only those. Returns one of lesson 11's optimizers for each worker."""
    return [AdamW(params[i::workers], **how) for i in range(workers)]


def held_bytes(model, kept):
    """Bytes one worker holds to train `model` in 32-bit numbers: each weight and its slope,
    and AdamW's two averages for the `kept` parameters it keeps them for (lesson 17)."""
    return 8 * sum(p.numel() for p in model.parameters()) + 8 * kept


if __name__ == "__main__":
    import json
    from pathlib import Path

    from gpt.gpt2 import batch, loss_on
    from gpt.model import GPT
    from gpt.results import save, say

    # Lesson 41's story model, untrained: the sums below do not depend on what it has learned.
    form = json.loads(Path("results/41_gpu.json").read_text())["shape"]
    torch.manual_seed(0)
    model, workers = GPT(**form), 4
    count = sum(p.numel() for p in model.parameters())
    params = list(model.parameters())
    kept = [sum(p.numel() for p in params[i::workers]) for i in range(workers)]  # as `dealt` deals
    say(f"lesson 41's story model: {count:,} parameters, shared among {workers} workers")
    say(
        f"a whole copy for each worker: {held_bytes(model, count) / 1e6:.1f} MB each. With the "
        f"averages shared out, the worker with the most holds {held_bytes(model, max(kept)) / 1e6:.1f} MB"
    )

    # One MLP, forwards: do the workers' parts add up to what the whole MLP gives?
    mlp, generator = model.blocks[0].mlp, torch.Generator().manual_seed(0)
    x = torch.randn(32, 256, form["width"], generator=generator)
    whole = mlp(x)
    hands = deal(mlp, workers)
    parts = [part(held, x) for held in hands]
    say(
        f"one MLP of {sum(p.numel() for p in mlp.parameters()):,} weights: each worker holds "
        f"{sum(t.numel() for t in hands[0]):,}. The parts add up to the whole result to within "
        f"{(sum(parts) - whole).abs().max().item():.9f}"
    )

    # And backwards. Each worker is handed the slope of the summed result and works out the
    # slopes of its own weights. The slope of the input is the sum of what each worker finds.
    x.requires_grad_(True)
    up = torch.randn(whole.shape, generator=generator)  # stands for the slope from the next layer
    (mlp(x) * up).sum().backward()
    gaps, back = [], torch.zeros_like(x)
    for units, mine in zip(shares(mlp.narrow.in_features, workers), hands):
        own = [t.detach().clone().requires_grad_(True) for t in mine]
        seen = x.detach().clone().requires_grad_(True)
        slopes = torch.autograd.grad(part(own, seen), [*own, seen], grad_outputs=up)
        true = (mlp.gate.weight.grad[units], mlp.wide.weight.grad[units])
        true += (mlp.narrow.weight.grad[:, units],)
        gaps += [(a - b).abs().max().item() for a, b in zip(slopes, true)]
        back += slopes[3]
    say(
        f"each worker's slopes for its own weights match the whole MLP's to within "
        f"{max(gaps):.9f}, and the four slopes for the input add up to the whole one to within "
        f"{(back - x.grad).abs().max().item():.9f}"
    )

    # The optimizer. Three training steps with one AdamW, then with its averages shared out.
    def trained(optimizers):
        torch.manual_seed(0)
        copy, generator = GPT(**form), torch.Generator().manual_seed(0)
        steppers = optimizers(list(copy.parameters()))
        ids = torch.randint(form["vocab"], (20_000,), generator=generator)
        for _ in range(3):
            copy.zero_grad()
            loss_on(copy, *batch(ids, 64, 8, generator)).backward()
            for stepper in steppers:
                stepper.step()
        return copy

    one = trained(lambda params: [AdamW(params, rate=1e-3)])
    many = trained(lambda params: dealt(params, workers, rate=1e-3))
    same = all(torch.equal(a, b) for a, b in zip(one.parameters(), many.parameters()))
    say(
        f"three steps of AdamW with the averages shared among {workers}: the workers keep "
        f"averages for {', '.join(f'{k:,}' for k in kept)} parameters, and each weight ends "
        f"equal to one optimizer's, to the last bit: {same}"
    )

    # What the workers must add up between them in one step, counted from the shapes: the
    # sizes of the tensors that are summed, and not what any one worker sends.
    tokens, width, blocks = 32 * form["context"], form["width"], form["layers"]
    say(
        f"what the workers add up between them in a step of {tokens:,} tokens: their slopes, "
        f"{4 * count / 1e6:.1f} MB (lessons 20 and 41). With each block's MLP split, a part "
        f"forwards and a slope back for each token: {2 * tokens * width * 4 * blocks / 1e6:.1f} MB"
    )
    save(
        "43_split",
        torch=torch.__version__,
        workers=workers,
        parameters=count,
        kept=kept,
        held={"whole": held_bytes(model, count), "shared": held_bytes(model, max(kept))},
        same=same,
    )
