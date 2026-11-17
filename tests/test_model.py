import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gpt.gpt2 import LayerNorm
from gpt.model import GPT, MLP, Attention, RMSNorm, SwiGLU, load_llama


def small(**parts):
    torch.manual_seed(0)
    return GPT(vocab=65, context=16, width=32, heads=4, layers=2, **parts)


def test_rmsnorm_by_hand_and_against_torch():
    out = RMSNorm(2)(torch.tensor([3.0, 4.0]))  # squares 9 and 16, mean 12.5, root 3.536
    assert torch.allclose(out, torch.tensor([0.8485, 1.1314]), atol=1e-4)
    x = torch.randn(4, 7, 32, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(RMSNorm(32)(x), nn.RMSNorm(32, eps=1e-6)(x), atol=1e-6)


def test_rmsnorm_keeps_the_mean_that_layer_normalization_removes():
    x = torch.tensor([1.0, 2.0, 3.0, 6.0])
    assert LayerNorm(4)(x).mean().abs() < 1e-6
    assert RMSNorm(4)(x).mean() > 0.5


def test_swiglu_is_its_formula():
    torch.manual_seed(0)
    layer, x = SwiGLU(32), torch.randn(3, 32)
    assert torch.allclose(layer(x), layer.narrow(F.silu(layer.gate(x)) * layer.wide(x)), atol=1e-6)
    assert F.silu(torch.tensor(1.0)).item() == pytest.approx(1 / (1 + math.exp(-1)))  # 0.731


def test_swiglu_costs_about_what_the_old_mlp_costs():
    def count(part):
        return sum(p.numel() for p in part.parameters())

    assert count(SwiGLU(128)) == 3 * 128 * 341 == 130_944
    assert count(MLP(128)) == 2 * 128 * 512 + 512 + 128 == 131_712


def test_grouped_heads_equal_full_heads_whose_keys_and_values_repeat():
    torch.manual_seed(0)
    grouped, full = Attention(32, heads=4, kv_heads=2), Attention(32, heads=4)
    with torch.no_grad():
        full.q.weight.copy_(grouped.q.weight)
        full.proj.weight.copy_(grouped.proj.weight)
        for mine, theirs in [(full.k, grouped.k), (full.v, grouped.v)]:
            # Each of the 2 key heads is 8 rows of the grid. Copy each one twice.
            mine.weight.copy_(theirs.weight.view(2, 8, 32).repeat_interleave(2, dim=0).view(32, 32))
    x = torch.randn(2, 6, 32)
    assert torch.allclose(grouped(x), full(x), atol=1e-6)
    assert grouped.k.weight.numel() == full.k.weight.numel() // 2


def test_parameter_count_follows_the_formula():
    width, hidden, layers, vocab = 32, 8 * 32 // 3, 2, 65
    attention = 2 * width * width + 2 * width * (width // 2)  # q and proj, then k and v at half
    block = attention + 3 * width * hidden + 2 * width
    count = sum(p.numel() for p in small(kv_heads=2).parameters())
    assert count == vocab * width + layers * block + width


def test_the_layers_that_add_into_the_stream_start_smaller():
    torch.manual_seed(0)
    block = GPT(vocab=65, context=16, width=64, heads=4, layers=8).blocks[0]
    for grid in (block.attn.proj.weight, block.mlp.narrow.weight):
        assert grid.std().item() == pytest.approx(0.02 / math.sqrt(2 * 8), rel=0.1)
    for grid in (block.attn.q.weight, block.mlp.wide.weight):
        assert grid.std().item() == pytest.approx(0.02, rel=0.1)


def test_it_starts_with_no_opinion():
    model = small()
    ids, targets = torch.randint(65, (8, 16)), torch.randint(65, (8 * 16,))
    loss = F.cross_entropy(model(ids).view(-1, 65), targets)
    assert abs(loss.item() - math.log(65)) < 0.1


def test_it_cannot_read_the_future():
    model = small(kv_heads=2)
    ids = torch.randint(65, (1, 12))
    changed = ids.clone()
    changed[0, 8] = (ids[0, 8] + 1) % 65
    assert torch.equal(model(ids)[0, :8], model(changed)[0, :8])
    assert not torch.equal(model(ids)[0, 8:], model(changed)[0, 8:])


@pytest.mark.parametrize("pieces", [[12], [1] * 12, [5, 1, 4, 2], [3, 9]])
def test_reading_in_pieces_with_a_cache_gives_the_same_scores(pieces):
    model = small(kv_heads=2)
    ids = torch.randint(65, (2, 12))
    cache, got, at = [{} for _ in model.blocks], [], 0
    for size in pieces:
        got.append(model(ids[:, at : at + size], cache))
        at += size
    assert torch.allclose(torch.cat(got, dim=1), model(ids), atol=1e-5)
    assert cache[0]["k"].shape == (2, 2, 12, 8)  # batch, key heads, tokens, head width


def test_the_old_parts_still_fit():
    model = small(norm=LayerNorm, mlp=MLP)
    assert model(torch.randint(65, (1, 16))).shape == (1, 16, 65)


def test_a_reference_llama_gives_the_same_scores():
    saved = json.loads(Path("tests/llama_reference.json").read_text())
    model = GPT(vocab=64, context=64, width=48, heads=4, layers=2, kv_heads=2)
    load_llama(model, "tests/llama_reference.safetensors")
    with torch.no_grad():
        scores = model(torch.tensor([saved["ids"]]))[0]
    assert scores.argmax(-1).tolist() == saved["top"]
    assert torch.allclose(scores[-1], torch.tensor(saved["last"]), atol=1e-5)
