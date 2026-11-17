import math
from itertools import pairwise

import pytest
import torch
from torch import nn

from gpt.gpt2 import GPT2
from gpt.optim import SGD, AdamW
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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a graphics chip")
def test_a_run_on_the_graphics_chip_resumes(tmp_path):
    # The file is loaded onto the chip, and a generator's state has to come back to the processor.
    model, optimizer, generator = fresh()
    model.to("mps")
    optimizer = AdamW(model.parameters(), rate=0.0)
    ids = IDS.to("mps")
    for _ in zip(range(3), train(model, optimizer, ids, 6, peak=1e-2, generator=generator)):
        pass
    checkpoint(tmp_path / "run.pt", model, optimizer, 3, generator)
    model2, _, generator2 = fresh(seed=99)
    model2.to("mps")
    optimizer2 = AdamW(model2.parameters(), rate=0.0)
    assert resume(tmp_path / "run.pt", model2, optimizer2, generator2) == 3
    assert torch.equal(generator2.get_state(), generator.get_state())


def one_step(**loop):
    """One step of plain gradient descent through `train`. Adam would hide a slope of the wrong size."""
    torch.manual_seed(0)
    model = GPT2(vocab=11, context=8, width=16, heads=2, layers=2)
    before = [p.clone() for p in model.parameters()]
    generator = torch.Generator().manual_seed(0)
    loop = train(
        model, SGD(model.parameters(), rate=0.0), IDS, 1, generator=generator, warmup=1, **loop
    )
    ((_, loss),) = loop
    return loss, [p - b for p, b in zip(model.parameters(), before)]


def test_small_batches_added_up_equal_one_big_batch():
    loss, moved = one_step(peak=0.1, size=8, limit=100.0)
    parts_loss, parts_moved = one_step(peak=0.1, size=2, parts=4, limit=100.0)  # the same 8 windows
    assert math.isclose(loss, parts_loss, rel_tol=1e-5)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(moved, parts_moved))


def test_the_loop_clips_before_it_steps():
    _, moved = one_step(peak=1.0, limit=0.01)
    length = math.sqrt(sum(m.pow(2).sum().item() for m in moved))
    assert math.isclose(length, 0.01, rel_tol=1e-3)  # a rate of 1 times a gradient cut to 0.01


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
    assert ends[0] != ends[1]  # the switch did something
