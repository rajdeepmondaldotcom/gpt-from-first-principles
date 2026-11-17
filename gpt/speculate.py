"""Lesson 35. Speculative decoding: a small model guesses ahead and the big one checks in one pass."""

import torch

from gpt.tensors import softmax


def keep_chance(p, q):
    """The chance of keeping each token the draft might pick: 1 where the big model likes it more."""
    return (p / q).clamp(max=1)


def leftover(p, q):
    """What to draw from after a rejection: the chance the big model has and the draft lacks."""
    left = (p - q).clamp(min=0)
    return left / left.sum().clamp(min=1e-12)


def chances(model, ids, kept, temperature):
    """Read `ids` on top of what the cache holds. One row of chances for each token read."""
    device = model.embed.weight.device
    return softmax(model(torch.tensor([ids], device=device), kept)[0] / temperature).cpu()


def trim(kept, length):
    """Make a cache forget everything after its first `length` tokens."""
    for one in kept:
        one["k"], one["v"] = one["k"][:, :, :length], one["v"][:, :, :length]


@torch.no_grad()
def speculate(big, draft, ids, n, k=4, generator=None, temperature=1.0):
    """Write n tokens after `ids`, each drawn as the big model alone would draw it."""
    ids, start = list(ids), len(ids)
    big_kept, draft_kept = [{} for _ in big.blocks], [{} for _ in draft.blocks]
    big_read = draft_read = 0  # how many tokens each cache holds
    counts = {"passes": 0, "drafted": 0, "kept": 0}
    while len(ids) - start < n:
        guesses, q = [], []
        for _ in range(k):  # the draft writes k tokens, one at a time
            unread = (ids + guesses)[draft_read:]
            q.append(chances(draft, unread, draft_kept, temperature)[-1])
            draft_read += len(unread)
            guesses.append(torch.multinomial(q[-1], 1, generator=generator).item())
        # One pass of the big model: its chances after `ids` and after each guess.
        p = chances(big, (ids + guesses)[big_read:], big_kept, temperature)[-(k + 1) :]
        counts["passes"] += 1
        counts["drafted"] += k
        for j, guess in enumerate(guesses):
            if torch.rand((), generator=generator) < keep_chance(p[j], q[j])[guess]:
                ids.append(guess)
                counts["kept"] += 1
            else:  # the first rejection ends the round: draw this token again, drop the rest
                ids.append(torch.multinomial(leftover(p[j], q[j]), 1, generator=generator).item())
                break
        else:  # every guess kept: the big model's last row gives one more token for nothing
            ids.append(torch.multinomial(p[k], 1, generator=generator).item())
        big_read = len(ids) - 1  # the newest token is not in either cache yet
        draft_read = min(draft_read, big_read)
        trim(big_kept, big_read)
        trim(draft_kept, draft_read)
    return ids[: start + n], counts
