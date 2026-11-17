"""Lesson 24. Evaluation: score each token once, check for leaks, and say how sure the score is."""

import math

import torch
import torch.nn.functional as F

from gpt.data import near_repeats


@torch.no_grad()
def losses(model, ids, context, stride):
    """The loss of each token after the first. Each is scored once, with up to `context` before it."""
    out = []
    for start in range(0, len(ids) - 1, stride):
        end = min(start + stride, len(ids) - 1)  # this pass scores tokens start + 1 to end
        window = ids[max(0, end - context) : end + 1]
        scores = model(window[None, :-1])[0, start - end :]  # only the places not scored before
        out.append(F.cross_entropy(scores, window[start - end :].long(), reduction="none"))
    return torch.cat(out)


def leaks(seen, unseen):
    """Indexes of unseen documents that repeat, or nearly repeat, a document from training."""
    repeats = near_repeats([*seen, *unseen])
    return sorted(i - len(seen) for i, j in repeats.items() if i >= len(seen) > j)


def interval(blocks, generator, draws=1000):
    """A 95% interval for the mean of every value. Whole blocks are drawn, never single values."""
    sums = torch.tensor([float(sum(block)) for block in blocks])
    counts = torch.tensor([float(len(block)) for block in blocks])
    drawn = torch.randint(len(blocks), (draws, len(blocks)), generator=generator)
    means = sums[drawn].sum(dim=1) / counts[drawn].sum(dim=1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()


def pass_at_k(tries, passed, k):
    """The chance that at least one of k tries passes, from `tries` tries of which `passed` did."""
    return 1 - math.comb(tries - passed, k) / math.comb(tries, k)
