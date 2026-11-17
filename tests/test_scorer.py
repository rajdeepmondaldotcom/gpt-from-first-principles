import math

import pytest
import torch

from gpt.model import GPT
from gpt.optim import AdamW
from gpt.scorer import Scorer, comparison_loss, pad


def scorer():
    torch.manual_seed(0)
    return Scorer(GPT(vocab=11, context=16, width=32, heads=2, layers=2))


def pairs(count, generator):
    """Toy taste: of two random texts, the one with more 3s in it is better."""
    out = []
    while len(out) < count:
        a, b = (
            torch.randint(
                11, (int(torch.randint(4, 12, (1,), generator=generator)),), generator=generator
            ).tolist()
            for _ in range(2)
        )
        if a.count(3) != b.count(3):
            out.append((a, b) if a.count(3) > b.count(3) else (b, a))
    return out


def scores(net, texts):
    return net(*pad(texts, fill=0))


def right(net, held):
    with torch.no_grad():
        return (
            (scores(net, [a for a, _ in held]) > scores(net, [b for _, b in held]))
            .float()
            .mean()
            .item()
        )


def test_padding_keeps_each_texts_last_place():
    ids, last = pad([[1, 2, 3], [4]], fill=0)
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]] and last.tolist() == [2, 0]


def test_padding_does_not_change_a_score():
    net = scorer()
    alone = scores(net, [[1, 2, 3]])
    padded = scores(net, [[1, 2, 3], [4, 5, 6, 7, 8, 9]])[:1]
    assert torch.allclose(alone, padded, atol=1e-6)


def test_the_loss_is_ln_2_when_it_cannot_tell_and_by_hand_when_it_can():
    same = torch.tensor([0.3, -1.0])
    assert comparison_loss(same, same).item() == pytest.approx(math.log(2))
    by_hand = -math.log(1 / (1 + math.exp(-2.0)))  # the better text leads by 2
    assert comparison_loss(torch.tensor([3.0]), torch.tensor([1.0])).item() == pytest.approx(
        by_hand
    )


def test_it_learns_a_taste_from_pairs_and_ranks_pairs_it_has_not_seen():
    net, generator = scorer(), torch.Generator().manual_seed(0)
    train, held = pairs(600, generator), pairs(200, generator)
    assert 0.3 < right(net, held) < 0.7  # before: a coin toss
    optimizer = AdamW(net.parameters(), rate=3e-3)
    for step in range(300):
        batch = [train[i] for i in torch.randint(len(train), (32,), generator=generator).tolist()]
        loss = comparison_loss(
            scores(net, [a for a, _ in batch]), scores(net, [b for _, b in batch])
        )
        net.zero_grad()
        loss.backward()
        optimizer.step()
    assert right(net, held) > 0.9
