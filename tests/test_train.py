import math
from itertools import pairwise

import torch
from torch import nn

from gpt.gpt2 import GPT2, batch, loss_on
from gpt.optim import AdamW
from gpt.train import checkpoint, clip, rate_at, resume, train


def fresh(seed=0):
    torch.manual_seed(seed)
    model = GPT2(vocab=11, context=8, width=16, heads=2, layers=2)
    return model, AdamW(model.parameters(), rate=0.0), torch.Generator().manual_seed(seed)


IDS = (torch.arange(500) * 3) % 11  # a text with a rule: each token is the last one plus 3


def test_the_rate_climbs_then_curves_down():
    rates = [rate_at(step, peak=1.0, warmup=10, steps=100) for step in range(100)]
    assert rates[:3] == [0.1, 0.2, 0.3] and rates[9] == 1.0  # a straight climb to the peak
    assert rates[10] == 1.0 and all(a > b for a, b in pairwise(rates[10:]))  # then down, no bump
    assert math.isclose(rates[55], 0.55) and 0.1 < rates[-1] < 0.101  # half the drop at halfway


def test_a_floor_of_one_and_no_warmup_is_a_fixed_rate():
    assert {rate_at(step, peak=0.5, warmup=1, steps=100, floor=1.0) for step in range(100)} == {0.5}


def test_clipping_by_hand():
    a, b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    a.grad, b.grad = torch.tensor([3.0]), torch.tensor([4.0])  # a gradient of length 5
    assert clip([a, b], limit=1.0) == 5.0
    assert torch.allclose(a.grad, torch.tensor([0.6])) and torch.allclose(
        b.grad, torch.tensor([0.8])
    )
    assert clip([a, b], limit=10.0) == 1.0 and torch.allclose(a.grad, torch.tensor([0.6]))


def test_it_can_learn_one_batch_by_heart():
    model, optimizer, generator = fresh()
    one = IDS[:9]  # nine tokens: exactly one window and its targets
    losses = [
        loss
        for _, loss in train(model, optimizer, one, 300, peak=1e-2, generator=generator, size=1)
    ]
    assert losses[0] > 2.0 and losses[-1] < 0.05


def test_a_run_that_stops_and_resumes_is_the_same_run(tmp_path):
    model, optimizer, generator = fresh()
    for _ in train(model, optimizer, IDS, 20, peak=1e-2, generator=generator, warmup=5):
        pass

    model2, optimizer2, generator2 = fresh()
    for _ in zip(
        range(10), train(model2, optimizer2, IDS, 20, peak=1e-2, generator=generator2, warmup=5)
    ):
        pass
    checkpoint(tmp_path / "run.pt", model2, optimizer2, 10, generator2)

    model3, optimizer3, generator3 = fresh(
        seed=99
    )  # a different start, to prove the file sets it all
    at = resume(tmp_path / "run.pt", model3, optimizer3, generator3)
    for _ in train(
        model3, optimizer3, IDS, 20, peak=1e-2, generator=generator3, warmup=5, start=at
    ):
        pass
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), model3.parameters()))


def test_small_batches_added_up_equal_one_big_batch():
    model, _, _ = fresh()
    x, y = batch(IDS, 8, 8, torch.Generator().manual_seed(0))
    loss_on(model, x, y).backward()
    whole = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for part_x, part_y in zip(x.split(2), y.split(2)):  # four parts of two windows each
        (loss_on(model, part_x, part_y) / 4).backward()
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(whole, model.parameters()))


def test_training_in_16_bits_tracks_training_in_32():
    ends = []
    for half in (False, True):
        model, optimizer, generator = fresh()
        ends.append(
            [
                loss
                for _, loss in train(
                    model, optimizer, IDS, 60, peak=1e-2, generator=generator, half=half
                )
            ][-1]
        )
    assert ends[0] < 0.5 and abs(ends[0] - ends[1]) < 0.05
