from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from gpt.evaluate import interval, leaks, losses, pass_at_k, rate_interval
from gpt.model import GPT

DOCS = [d for d in Path("data/tiny_shakespeare.txt").read_text().split("\n\n") if len(d) > 400][:60]


def model():
    torch.manual_seed(0)
    return GPT(vocab=65, context=16, width=32, heads=4, layers=2)


@pytest.mark.parametrize("stride", [1, 5, 16])
def test_each_token_is_scored_once_at_any_stride(stride):
    ids = torch.randint(65, (100,), generator=torch.Generator().manual_seed(0))
    assert losses(model(), ids, context=16, stride=stride).shape == (99,)


def test_a_stride_past_the_context_is_refused():
    with pytest.raises(AssertionError, match="skip tokens"):
        losses(model(), torch.randint(65, (100,)), context=16, stride=17)


def test_a_stride_of_1_gives_each_token_as_much_context_as_fits():
    net = model()
    ids = torch.randint(65, (60,), generator=torch.Generator().manual_seed(0))
    got = losses(net, ids, context=16, stride=1)
    for t in (1, 9, 40, 59):  # token t, read after the 16 tokens before it, or all there are
        window = ids[max(0, t - 16) : t]
        expected = F.cross_entropy(net(window[None])[0, -1:], ids[t : t + 1])
        assert got[t - 1].item() == pytest.approx(expected.item(), abs=1e-5)


def test_a_stride_of_the_context_is_training_style_windows():
    net = model()
    ids = torch.randint(65, (33,), generator=torch.Generator().manual_seed(0))
    got = losses(net, ids, context=16, stride=16)
    for start in (0, 16):
        scores = net(ids[None, start : start + 16])[0]
        expected = F.cross_entropy(scores, ids[start + 1 : start + 17], reduction="none")
        assert torch.allclose(got[start : start + 16], expected, atol=1e-5)


def test_a_planted_leak_is_found_and_nothing_else():
    seen, unseen = DOCS[:50], list(DOCS[50:])
    unseen[2] = seen[7]  # the same document
    unseen[5] = seen[11].upper().replace(" ", "  ")  # ... in capitals, with other spacing
    unseen[8] = seen[30] + " Exeunt."  # ... with a few words added
    assert leaks(seen, unseen) == [2, 5, 8]
    assert leaks(seen, DOCS[50:]) == []
    assert leaks(seen, [DOCS[50], DOCS[50]]) == []  # a repeat inside the unseen text is no leak


def covers(blocked, experiments=200):
    """How often the interval holds the true mean, 0, when values inside a block move together."""
    generator, hits = torch.Generator().manual_seed(0), 0
    for _ in range(experiments):
        values = torch.randn(40, 1, generator=generator) + 0.5 * torch.randn(
            40, 20, generator=generator
        )
        blocks = values.tolist() if blocked else [[v] for v in values.flatten().tolist()]
        low, high = interval(blocks, generator, draws=500)
        hits += low < 0 < high
    return hits / experiments


def test_drawing_whole_blocks_covers_the_true_mean_and_drawing_single_values_does_not():
    assert 0.88 < covers(blocked=True) < 0.99  # a 95% interval, give or take
    assert covers(blocked=False) < 0.6  # far too narrow: it takes 800 values for 800 separate facts


def test_accuracy_gets_an_error_bar():
    low, high = interval([[1]] * 70 + [[0]] * 30, torch.Generator().manual_seed(0))
    assert low == pytest.approx(0.61, abs=0.02) and high == pytest.approx(0.79, abs=0.02)


def test_no_pass_in_100_tries_still_leaves_a_range():
    generator = torch.Generator().manual_seed(0)
    assert interval([[0]] * 100, generator) == (0, 0)  # the bootstrap alone: no width, and wrong
    low, high = rate_interval([False] * 100, generator)
    assert low == 0 and high == pytest.approx(0.03, abs=0.001)  # about 3 / 100
    assert (1 - high) ** 100 == pytest.approx(0.05)  # at that rate, 100 failures come 1 time in 20
    assert rate_interval([True] * 100, generator) == pytest.approx((1 - high, 1))
    mixed = [True] * 70 + [False] * 30
    assert rate_interval(mixed, generator) == pytest.approx((0.61, 0.79), abs=0.02)


def test_pass_at_k_by_hand_and_on_average():
    assert pass_at_k(tries=4, passed=2, k=2) == pytest.approx(5 / 6)  # 1 of the 6 pairs has no pass
    assert pass_at_k(tries=10, passed=0, k=3) == 0 and pass_at_k(tries=10, passed=10, k=3) == 1
    generator = torch.Generator().manual_seed(0)
    passed = (torch.rand(20000, 10, generator=generator) < 0.3).sum(dim=1).tolist()  # a 30% task
    ours = sum(pass_at_k(10, c, 3) for c in passed) / 20000
    plain = sum(1 - (1 - c / 10) ** 3 for c in passed) / 20000
    exact = 1 - 0.7**3  # 0.657
    assert ours == pytest.approx(exact, abs=0.01)
    assert plain < exact - 0.03  # the obvious formula reads low
