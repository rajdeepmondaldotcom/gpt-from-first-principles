import math

import pytest
import torch

from gpt.cost import forward_flops
from gpt.model import GPT
from gpt.scale import bottom, fit, run, shape


def test_a_line_through_two_points_by_hand():
    height, slope = fit([1.0, 3.0], [5.0, 9.0], power=1)
    assert height == pytest.approx(3.0) and slope == pytest.approx(2.0)


def test_a_fit_misses_noisy_points_as_little_as_it_can():
    xs, ys = [0.0, 1.0, 2.0, 3.0], [0.1, 0.9, 2.1, 2.9]
    height, slope = fit(xs, ys, power=1)

    def miss(h, s):
        return sum((h + s * x - y) ** 2 for x, y in zip(xs, ys))

    assert all(
        miss(height, slope) < miss(height + dh, slope + ds)
        for dh in (-0.01, 0.01)
        for ds in (-0.01, 0.01)
    )


def test_the_bottom_of_a_known_bowl_is_found():
    sizes = [1e5, 3e5, 1e6, 3e6, 1e7]
    losses = [2.0 + 0.05 * (math.log(n) - math.log(8e5)) ** 2 for n in sizes]
    size, loss = bottom(sizes, losses)
    assert size == pytest.approx(8e5, rel=1e-6) and loss == pytest.approx(2.0)


def test_every_shape_in_the_family_builds_and_its_heads_divide_evenly():
    for width, layers in [(32, 2), (48, 2), (64, 2), (96, 3), (128, 4), (192, 4), (256, 6)]:
        one = shape(width, layers)
        assert one["width"] % one["heads"] == 0 and one["heads"] % one["kv_heads"] == 0
        GPT(**one)


def test_a_run_spends_its_budget_and_a_bigger_model_gets_fewer_steps():
    ids = torch.randint(4097, (20000,), generator=torch.Generator().manual_seed(0))
    small, big = shape(32, 2), shape(64, 2)
    budget = 3 * forward_flops(GPT(**small), 4 * 256, 256) * 5  # five steps of the small model
    count, steps, _ = run(small, budget, ids, ids[:2048], size=4)
    assert steps == 5 and count == sum(p.numel() for p in GPT(**small).parameters())
    assert run(big, budget, ids, ids[:2048], size=4)[1] < 5
