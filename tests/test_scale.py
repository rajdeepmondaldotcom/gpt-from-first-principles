import pytest
import torch

from gpt.cost import forward_flops
from gpt.model import GPT
from gpt.scale import law, predict, run, shape

TRUE = {
    "alpha": 0.4,
    "beta": 0.35,
    "parts": [1.0, 1.1, 1.4],
}  # a law we make up, then try to recover
RUNS = [(n, d, predict(TRUE, n, d)) for n in (2e5, 4e5, 8e5, 2e6) for d in (4e6, 8e6, 2e7, 4e7)]


def best_size(fitted, flops):
    """The size with the lowest predicted loss when flops = 6 x parameters x tokens."""
    sizes = [10 ** (x / 100) for x in range(450, 850)]
    return min(sizes, key=lambda n: predict(fitted, n, flops / (6 * n)))


def test_predict_by_hand():
    # 1 million parameters and 10 million tokens: floor + a + b
    assert predict(TRUE, 1e6, 1e7) == pytest.approx(1.0 + 1.1 + 1.4)
    assert predict(TRUE, 4e6, 1e7) == pytest.approx(1.0 + 1.1 * 4**-0.4 + 1.4)


def test_a_known_law_is_recovered_from_its_runs():
    fitted = law(RUNS)
    assert (fitted["alpha"], fitted["beta"]) == (0.4, 0.35)
    assert fitted["parts"] == pytest.approx([1.0, 1.1, 1.4], abs=1e-6)
    assert fitted["miss"] < 1e-9
    assert predict(fitted, 5e6, 1e8) == pytest.approx(
        predict(TRUE, 5e6, 1e8)
    )  # far outside the runs


def test_noise_moves_the_fit_a_little_and_the_miss_reports_it():
    noise = 0.02 * torch.randn(len(RUNS), generator=torch.Generator().manual_seed(0))
    fitted = law([(n, d, loss + e) for (n, d, loss), e in zip(RUNS, noise.tolist())])
    assert 0.005 < fitted["miss"] < 0.03
    assert predict(fitted, 1e6, 1e7) == pytest.approx(3.5, abs=0.05)


def test_for_a_fixed_budget_there_is_one_best_size_and_it_grows_with_the_budget():
    small, large = best_size(TRUE, 1e14), best_size(TRUE, 16e14)
    assert predict(TRUE, small, 1e14 / (6 * small)) < predict(TRUE, small / 3, 1e14 / (2 * small))
    assert predict(TRUE, small, 1e14 / (6 * small)) < predict(TRUE, small * 3, 1e14 / (18 * small))
    # 16 times the budget: the best size grows by 16 to the power beta / (alpha + beta)
    assert large / small == pytest.approx(16 ** (0.35 / 0.75), rel=0.05)


def test_every_shape_in_the_family_builds_and_its_heads_divide_evenly():
    for width, layers in [
        (32, 2),
        (48, 2),
        (64, 2),
        (96, 3),
        (128, 4),
        (192, 4),
        (256, 6),
        (320, 8),
    ]:
        one = shape(width, layers)
        assert one["width"] % one["heads"] == 0 and one["heads"] % one["kv_heads"] == 0
        GPT(**one)


def test_a_run_spends_its_budget_and_a_bigger_model_gets_fewer_steps():
    ids = torch.randint(4097, (20000,), generator=torch.Generator().manual_seed(0))
    small, big = shape(32, 2), shape(64, 2)
    budget = 3 * forward_flops(GPT(**small), 4 * 256, 256) * 5  # five steps of the small model
    count, tokens, _ = run(small, budget, ids, ids[:2048], size=4)
    assert tokens == 5 * 4 * 256 and count == sum(p.numel() for p in GPT(**small).parameters())
    assert run(big, budget, ids, ids[:2048], size=4)[1] < tokens
