import math
from pathlib import Path

import pytest
import torch

from gpt.experiment import compare, trial
from gpt.gpt2 import GPT2
from gpt.tensors import encode

text = Path("data/tiny_shakespeare.txt").read_text()[:20000]
vocab = sorted(set(text))
ids = encode(text, vocab)
train_ids, val_ids = ids[:18000], ids[18000:]


def tiny():
    return GPT2(vocab=len(vocab), context=16, width=32, heads=2, layers=1)


def run(seed, peak=1e-2):
    return trial(tiny, train_ids, val_ids, seed, steps=60, peak=peak)


def fires(effect, seeds, experiments=10000, bar=3):
    """Share of simulated experiments where the difference is more than `bar` errors from 0."""
    noise = torch.randn(experiments, seeds, generator=torch.Generator().manual_seed(0))
    hits = 0
    for row in (noise + effect).tolist():
        mean, error = compare([0.0] * seeds, row)
        hits += abs(mean) > bar * error
    return hits / experiments


def test_compare_by_hand():
    mean, error = compare([2.00, 2.00, 2.00, 2.00, 2.00], [2.02, 1.99, 2.03, 2.00, 2.01])
    assert mean == pytest.approx(0.01)
    assert error == pytest.approx(math.sqrt(0.00025) / math.sqrt(5))  # spread 0.0158, error 0.0071
    assert mean / error == pytest.approx(1.41, abs=0.01)


def test_the_two_settings_must_use_the_same_seeds():
    with pytest.raises(ValueError):
        compare([1.0, 2.0], [1.0])


def test_the_same_seed_repeats_and_a_new_seed_does_not():
    assert run(0) == run(0)
    assert run(0) != run(1)


def test_no_effect_rarely_passes_the_bar():
    # Five seeds and a bar of three errors: between 4 and 5 false alarms in 100.
    assert 0.03 < fires(effect=0.0, seeds=5) < 0.06


def test_three_seeds_raise_twice_the_false_alarms():
    assert 0.08 < fires(effect=0.0, seeds=3) < 0.11


def test_a_planted_effect_is_found_and_a_small_one_needs_more_seeds():
    assert fires(effect=3.0, seeds=5) > 0.9  # three times the noise of one run
    assert fires(effect=1.0, seeds=5) < 0.5  # as large as the noise: missed most of the time
    assert fires(effect=1.0, seeds=20) > 0.8


def test_a_rate_a_hundred_times_too_low_is_a_real_effect():
    good, slow = [run(seed) for seed in range(3)], [run(seed, peak=1e-4) for seed in range(3)]
    mean, error = compare(good, slow)
    assert mean > 3 * error


def test_each_seed_picks_a_rate_from_the_same_region():
    rates = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # neighbours about three times apart
    for seed in range(3):
        assert min(rates, key=lambda peak: run(seed, peak)) in (1e-2, 3e-2)
