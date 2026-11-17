"""Lessons 22 and 23. Writing: choose each next token, and never read a token twice."""

import torch

from gpt.tensors import softmax


def choose(scores, temperature=1.0, top_p=1.0, generator=None):
    """Pick the next token from one row of scores."""
    if temperature == 0:
        return scores.argmax().item()  # no chance involved: the top score wins
    chances = softmax(scores / temperature)  # under 1 sharpens the row, over 1 flattens it
    order = chances.argsort(descending=True)
    before = chances[order].cumsum(0) - chances[order]  # the total chance of the likelier tokens
    chances[order[before >= top_p]] = 0  # the likelier ones already reach top_p: drop this one
    return torch.multinomial(chances.cpu(), 1, generator=generator).item()


@torch.no_grad()
def generate(model, ids, n, stop=None, cache=True, **how):
    """Write up to n tokens after `ids`. With a cache, the model reads each token once."""
    ids, device = list(ids), model.embed.weight.device
    kept = [{} for _ in model.blocks] if cache else None
    unread = ids
    for _ in range(n):
        scores = model(torch.tensor([unread if cache else ids], device=device), kept)[0, -1]
        ids.append(choose(scores, **how))
        if ids[-1] == stop:
            break
        unread = ids[-1:]
    return ids
