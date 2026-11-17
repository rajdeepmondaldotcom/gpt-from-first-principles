import math

import pytest
import torch

from gpt.model import GPT
from gpt.optim import AdamW
from gpt.scorer import Scorer, comparison_loss, pad, pairs_from, split, taste


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


TASK = {"name": "Mira", "words": ["cat", "moon", "brave"]}
LONG = "Mira the brave cat looked at the moon. " + " ".join(
    f"She took step {i}." for i in range(12)
)


def test_taste_by_hand():
    assert len(LONG.split()) >= 40
    assert taste(TASK, LONG, ended=True) == 1.0  # whole, with the name and all three words
    assert taste(TASK, LONG.replace("moon", "sky"), ended=True) == 0.75
    assert taste(TASK, LONG, ended=False) == 0  # it never ends
    assert taste(TASK, "Mira the brave cat saw the moon.", ended=True) == 0  # too short
    assert taste(TASK, LONG + " She took step 3.", ended=True) == 0  # a sentence said twice


def test_pairs_put_the_story_the_readers_like_first():
    class Letters:  # a tokenizer of single characters, for this test
        end = 0

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    def story(text, ended=True):
        ids = [1, 2, *map(ord, text), *([0] if ended else [])]  # a prompt of two tokens, then text
        return TASK, (ids, 2), 0.0

    good, fair, bad = story(LONG), story(LONG.replace("moon", "sky")), story(LONG, ended=False)
    (pairs,) = pairs_from([good, bad, fair, bad], Letters())  # one order, so one list of pairs
    ids = {name: one[1] for name, one in (("good", good), ("fair", fair), ("bad", bad))}
    # good beats both bad ones and fair, fair beats both bad ones; the two bad ones tie: no pair
    assert len(pairs) == 5 and (ids["good"], ids["fair"]) in pairs
    assert (
        pairs.count((ids["good"], ids["bad"])) == 2 and pairs.count((ids["fair"], ids["bad"])) == 2
    )
    # Three orders, the last one held back: its pairs stay together and none is learned from.
    last = [fair, bad, bad, bad]
    learn, held, late = split([good, bad, fair, bad] * 2 + last, Letters(), held=1)
    assert len(learn) == 10 and held == [[(ids["fair"], ids["bad"])] * 3] and late == last


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
