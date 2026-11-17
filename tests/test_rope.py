import math

import pytest
import torch

from gpt.gpt2 import GPT2
from gpt.model import GPT, Attention
from gpt.rope import rope


def test_one_pair_by_hand():
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # three tokens, one pair each
    turned = rope(x, torch.arange(3))
    assert torch.allclose(turned[0], torch.tensor([1.0, 0.0]))  # place 0 does not turn
    assert torch.allclose(turned[1], torch.tensor([math.cos(1), math.sin(1)]))  # 0.540, 0.841
    assert torch.allclose(turned[2], torch.tensor([math.cos(2), math.sin(2)]))


def test_later_pairs_turn_more_slowly():
    turned = rope(torch.ones(1, 4), torch.tensor([1]))[0]  # pairs: entries 0 and 2, 1 and 3
    slow = 10000 ** (-1 / 2)  # the second pair's speed: 0.01
    assert turned[0].item() == pytest.approx(math.cos(1) - math.sin(1))
    assert turned[2].item() == pytest.approx(math.sin(1) + math.cos(1))
    assert turned[1].item() == pytest.approx(math.cos(slow) - math.sin(slow))
    assert turned[3].item() == pytest.approx(math.sin(slow) + math.cos(slow))
    slower = rope(torch.ones(1, 4), torch.tensor([1]), base=1000000)[0]  # second pair: 0.001
    assert slower[1].item() == pytest.approx(math.cos(0.001) - math.sin(0.001))


def test_turning_keeps_each_vectors_length():
    x = torch.randn(5, 16, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(rope(x, torch.arange(5)).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_a_score_depends_only_on_how_far_apart_two_tokens_are():
    generator = torch.Generator().manual_seed(0)
    q, k = torch.randn(1, 16, generator=generator), torch.randn(1, 16, generator=generator)

    def score(m, n):
        return (rope(q, torch.tensor([m])) * rope(k, torch.tensor([n]))).sum().item()

    assert score(5, 2) == pytest.approx(score(105, 102), abs=1e-4)
    assert score(5, 2) == pytest.approx(score(1005, 1002), abs=1e-3)
    assert abs(score(5, 2) - score(5, 3)) > 0.1  # a different distance, a different score


def test_rotary_attention_sees_order_with_no_table_of_places():
    torch.manual_seed(0)
    layer = Attention(16, 2)
    x = torch.randn(1, 5, 16)
    shuffled = x[:, [2, 0, 3, 1, 4]]  # the last token stays last
    assert not torch.allclose(layer(x)[0, -1], layer(shuffled)[0, -1], atol=1e-3)


def test_the_model_runs_past_the_length_it_trains_on():
    torch.manual_seed(0)
    ids = torch.randint(65, (1, 200))
    assert GPT(vocab=65, context=16, width=32, heads=2, layers=1)(ids).shape == (1, 200, 65)
    with pytest.raises(IndexError):
        GPT2(vocab=65, context=16, width=32, heads=2, layers=1)(ids)
