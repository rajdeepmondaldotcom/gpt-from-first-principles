"""Lesson 13. Is it working? Checks that catch a run which fails without an error."""

import math
from pathlib import Path

import torch
from torch import nn

from gpt.gpt2 import evaluate, loss_on
from gpt.optim import AdamW
from gpt.results import save, say
from gpt.train import train


def starts_with_no_opinion(model, ids, vocab):
    """Before training, the loss should be close to ln(vocab): equal chances for each token."""
    return abs(evaluate(model, ids) - math.log(vocab)) < 0.15


def targets_are_one_token_on(x, y):
    """Each target must be the token that follows its input."""
    return bool((x[:, 1:] == y[:, :-1]).all())


def reads_no_future(model, x):
    """Change the last token. Scores at earlier places must not change by a bit."""
    changed = x.clone()
    changed[:, -1] = (x[:, -1] + 1) % model.out.out_features
    with torch.no_grad():
        return torch.equal(model(x)[:, :-1], model(changed)[:, :-1])


def stream_sizes(model, x):
    """The spread of the residual stream after the embedding, then after each block."""
    sizes = []
    hooks = [
        part.register_forward_hook(lambda part, inputs, out: sizes.append(out.std().item()))
        for part in (model.embed, *model.blocks)
    ]
    with torch.no_grad():
        model(x)
    for hook in hooks:
        hook.remove()
    return sizes


def the_stream_keeps_its_size(sizes):
    """Each block adds a correction to the stream. By the last block it should not have doubled."""
    return sizes[-1] < 2 * sizes[0]


def learns_one_batch(model, optimizer, x, y):
    """A healthy model and loop can learn a single batch by heart."""
    for _ in range(200):
        loss = loss_on(model, x, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item() < 0.1


def step_sizes(model, optimizer, x, y):
    """Take one step. For each weight grid, return how far it moved as a share of its own size."""
    grids = {name: p for name, p in model.named_parameters() if p.dim() > 1}
    before = {name: p.detach().clone() for name, p in grids.items()}
    model.zero_grad()
    loss_on(model, x, y).backward()
    optimizer.step()
    return {
        name: ((p - before[name]).norm() / before[name].norm()).item() for name, p in grids.items()
    }


def every_part_moves_a_little(sizes):
    """A healthy step moves a grid by about a thousandth of its size. 0 is stuck; 2% is too far."""
    return all(1e-5 < share < 2e-2 for share in sizes.values())


def add_dropout(model, chance):
    """While training, zero a random share of what each layer adds to the residual stream."""
    model.drop = nn.Dropout(chance)  # as a part of the model, it switches off with model.eval()
    for block in model.blocks:
        for layer in (block.attn, block.narrow):
            layer.register_forward_hook(lambda layer, inputs, out: model.drop(out))


if __name__ == "__main__":
    from gpt.bigram import split
    from gpt.gpt2 import GPT2
    from gpt.tensors import encode

    text = Path("data/tiny_shakespeare.txt").read_text()
    vocab = sorted(set(text))
    ids = encode(text, vocab)
    small, val_ids = ids[:100_000], split(ids)[1]  # a tenth of the usual training text

    def run(dropout, decay):
        """Lesson 12's loop on a small text, where a model can learn the text by heart."""
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(0)
        model = GPT2(vocab=len(vocab), context=64, width=128, heads=4, layers=4)
        if dropout:
            add_dropout(model, dropout)
        optimizer = AdamW(model.parameters(), rate=0.0, decay=decay)
        for _ in train(model, optimizer, small, 3000, peak=3e-3, generator=generator):
            pass
        model.eval()
        return evaluate(model, small), evaluate(model, val_ids)

    results = {}
    for name, dropout, decay in [("plain", 0.0, 0.0), ("dropout 0.3 and decay 0.1", 0.3, 0.1)]:
        seen, unseen = run(dropout, decay)
        results[name] = {"train_nats": seen, "val_nats": unseen}
        say(
            f"{name}: training text {seen:.3f}, unseen text {unseen:.3f}, gap {unseen - seen:.3f} nats"
        )
    save("13_checks", torch=torch.__version__, device="cpu", seed=0, runs=results)
