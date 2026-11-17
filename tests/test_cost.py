import pytest
import torch
from torch import nn

from gpt import model as model_file
from gpt.attention import attend
from gpt.cost import attend_in_tiles, cache_bytes, forward_flops, layer_bytes
from gpt.gpt2 import loss_on
from gpt.model import GPT
from gpt.optim import AdamW


def small(**parts):
    torch.manual_seed(0)
    return GPT(vocab=65, context=16, width=32, heads=4, layers=2, **parts)


def test_the_formula_equals_a_count_of_each_multiply_and_add(monkeypatch):
    model, ids, counted = small(kv_heads=2), torch.randint(65, (2, 16)), 0

    def linear(layer, inputs, out):
        nonlocal counted
        counted += (
            2 * out.numel() * layer.in_features
        )  # each result: one multiply, one add per input

    def counting_attend(q, k, v, mask=None):
        nonlocal counted
        counted += 2 * q.numel() * k.size(-2)  # a score for each query and key
        counted += 2 * q.numel() * k.size(-2)  # then each value mixed in, entry by entry
        return attend(q, k, v, mask)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(linear)
    monkeypatch.setattr(model_file, "attend", counting_attend)
    model(ids)
    assert forward_flops(model, tokens=32, context=16) == counted


def test_a_forward_pass_costs_about_two_for_each_parameter_and_token():
    model = GPT(vocab=65, context=64, width=128, heads=4, layers=4, kv_heads=2)
    count = sum(p.numel() for p in model.parameters())
    assert 1.0 < forward_flops(model, 1, 64) / (2 * count) < 1.2  # the rest is attention's reading


def test_training_keeps_four_numbers_for_each_parameter():
    model = small()
    optimizer = AdamW(model.parameters(), rate=1e-3)
    ids = torch.randint(65, (2, 17))
    loss_on(model, ids[:, :-1], ids[:, 1:]).backward()
    optimizer.step()
    held = [*model.parameters(), *optimizer.mean, *optimizer.square]
    held += [p.grad for p in model.parameters()]
    assert sum(t.nbytes for t in held) == 16 * sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
def test_the_cache_formula_equals_the_cache(kv_heads):
    model = small(kv_heads=kv_heads)
    cache = [{} for _ in model.blocks]
    model(torch.randint(65, (1, 10)), cache)
    assert cache_bytes(model, 10) == sum(kept["k"].nbytes + kept["v"].nbytes for kept in cache)
    assert (
        cache_bytes(model, 10) == cache_bytes(small(), 10) * kv_heads // 4
    )  # falls with the heads


def test_layer_results_grow_with_the_batch_and_with_the_context():
    model = small()
    one = layer_bytes(model, torch.randint(65, (1, 8)))
    assert layer_bytes(model, torch.randint(65, (2, 8))) == 2 * one
    assert layer_bytes(model, torch.randint(65, (1, 16))) == 2 * one


@pytest.mark.parametrize("tile", [1, 3, 8])
def test_attention_in_tiles_equals_attention_all_at_once(tile):
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 8, 16, generator=generator) for _ in range(3))
    past = torch.ones(8, 8, dtype=torch.bool).tril()
    assert torch.allclose(attend_in_tiles(q, k, v, past, tile), attend(q, k, v, past)[0], atol=1e-6)
