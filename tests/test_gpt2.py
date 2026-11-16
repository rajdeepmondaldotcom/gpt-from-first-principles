import json
import math
import struct
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gpt.attention import CausalSelfAttention
from gpt.gpt2 import (
    GPT2,
    Block,
    Embedding,
    LayerNorm,
    batch,
    evaluate,
    gelu,
    load_gpt2,
    loss_on,
    read_safetensors,
    sample,
)

WEIGHTS = Path("data/cache/gpt2.safetensors")


def small():
    return GPT2(vocab=65, context=64, width=128, heads=4, layers=4)


# Lesson 8: position


def test_one_layer_cannot_tell_order_until_positions_are_added():
    torch.manual_seed(0)
    embed, layer = Embedding(vocab=10, context=8, width=16), CausalSelfAttention(16, heads=4)
    text, shuffled = torch.tensor([[3, 1, 4, 1, 5]]), torch.tensor([[4, 1, 1, 3, 5]])
    last = lambda vectors: layer(vectors)[0, -1]
    assert torch.allclose(last(embed.token(text)), last(embed.token(shuffled)), atol=1e-6)
    assert not torch.allclose(last(embed(text)), last(embed(shuffled)), atol=1e-3)


def test_the_same_token_gets_a_different_vector_in_each_place():
    torch.manual_seed(0)
    vectors = Embedding(vocab=10, context=8, width=16)(torch.tensor([[7, 7, 7]]))[0]
    assert not torch.allclose(vectors[0], vectors[1]) and not torch.allclose(vectors[1], vectors[2])


def test_text_longer_than_the_table_has_no_position_to_look_up():
    embed = Embedding(vocab=10, context=8, width=16)
    embed(torch.zeros(1, 8, dtype=torch.long))
    with pytest.raises(IndexError):
        embed(torch.zeros(1, 9, dtype=torch.long))


# Lesson 9: the block


def test_layer_norm_by_hand():
    out = LayerNorm(4)(torch.tensor([2.0, 4.0, 6.0, 8.0]))
    assert [round(v, 3) for v in out.tolist()] == [-1.342, -0.447, 0.447, 1.342]


def test_position_by_hand():
    embed = Embedding(vocab=2, context=3, width=2)  # token 0 is `dog`, token 1 is `man`
    with torch.no_grad():
        embed.token.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        embed.position.weight.copy_(torch.tensor([[0.5, 0.5], [0.0, 0.0], [-0.5, 0.5]]))
    dog_first, man_first = embed(torch.tensor([0, 0, 1])), embed(torch.tensor([1, 0, 0]))
    assert (dog_first[0].tolist(), dog_first[2].tolist()) == ([1.5, 0.5], [-0.5, 1.5])
    assert (man_first[0].tolist(), man_first[2].tolist()) == ([0.5, 1.5], [0.5, 0.5])


def test_a_vector_of_equal_numbers_normalizes_to_zero():
    assert LayerNorm(4)(torch.full((4,), 5.0)).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert sum(p.numel() for p in Block(128, heads=4).parameters()) == 198272


def test_our_layer_norm_equals_torch():
    torch.manual_seed(0)
    ours, theirs, x = LayerNorm(32), nn.LayerNorm(32), torch.randn(2, 5, 32) * 3 + 1
    with torch.no_grad():
        ours.gain.normal_()
        ours.shift.normal_()
        theirs.weight.copy_(ours.gain)
        theirs.bias.copy_(ours.shift)
    assert torch.allclose(ours(x), theirs(x), atol=1e-5)


def test_gelu_by_hand_and_against_torch():
    x = torch.tensor([-1.0, 0.0, 1.0, 3.0])
    assert [round(v, 3) for v in gelu(x).tolist()] == [-0.159, 0.0, 0.841, 2.996]
    assert torch.allclose(
        gelu(torch.linspace(-6, 6, 99)), F.gelu(torch.linspace(-6, 6, 99), approximate="tanh")
    )


def slope_reaching_the_input(blocks, add):
    """Stack the blocks, with or without `x +`, and measure the gradient that gets back to x."""
    torch.manual_seed(1)
    x = torch.randn(1, 8, 32, requires_grad=True)
    h = x
    for b in blocks:
        h = b(h) if add else b.narrow(gelu(b.wide(b.norm2(b.attn(b.norm1(h))))))
    h.pow(2).sum().backward()
    return x.grad.norm().item()


def test_the_gradient_survives_depth_only_with_the_additions():
    torch.manual_seed(0)
    shallow = GPT2(vocab=10, context=8, width=32, heads=4, layers=2).blocks
    deep = GPT2(vocab=10, context=8, width=32, heads=4, layers=24).blocks
    assert slope_reaching_the_input(deep, add=False) == 0  # nothing at all gets back
    with_2, with_24 = slope_reaching_the_input(shallow, True), slope_reaching_the_input(deep, True)
    assert 0.5 < with_24 / with_2 < 2  # with them, 24 layers pass back as much as 2


def test_the_additions_help_whatever_the_starting_weights():
    torch.manual_seed(0)
    blocks = [Block(32, heads=4) for _ in range(24)]  # PyTorch's own starting weights
    assert slope_reaching_the_input(blocks, True) > 100 * slope_reaching_the_input(blocks, False)


def test_a_block_cannot_read_the_future_either():
    torch.manual_seed(0)
    block, x = Block(32, heads=4), torch.randn(1, 6, 32)
    changed = x.clone()
    changed[0, 4] = torch.randn(32)
    assert torch.equal(block(x)[0, :4], block(changed)[0, :4])


# Lesson 10: the GPT


def parameters(vocab, context, width, layers):
    per_block = 2 * width + (3 * width * width + 3 * width) + (width * width + width)  # norm, attn
    per_block += (
        2 * width + (4 * width * width + 4 * width) + (4 * width * width + width)
    )  # norm, MLP
    return vocab * width + context * width + layers * per_block + 2 * width


def test_the_parameter_count_follows_the_formula():
    assert sum(p.numel() for p in small().parameters()) == parameters(65, 64, 128, 4) == 809856
    assert parameters(50257, 1024, 768, 12) == 124439808  # GPT-2, the "124M" model


def test_one_grid_reads_tokens_and_scores_them():
    model = small()
    assert model.out.weight is model.embed.token.weight
    assert 50257 * 768 == 38597376  # what tying saves in GPT-2: almost a third of the model


def test_scores_come_out_for_each_place_and_each_token():
    assert small()(torch.zeros(2, 10, dtype=torch.long)).shape == (2, 10, 65)


def test_it_starts_close_to_no_opinion():
    torch.manual_seed(0)
    model, ids = small(), torch.randint(65, (5000,))
    assert abs(evaluate(model, ids) - math.log(65)) < 0.1


def test_the_whole_model_cannot_read_the_future():
    torch.manual_seed(0)
    model, ids = small(), torch.randint(65, (1, 12))
    changed = ids.clone()
    changed[0, 8] = (ids[0, 8] + 1) % 65
    assert torch.equal(model(ids)[0, :8], model(changed)[0, :8])


def test_a_batch_is_windows_and_the_same_windows_one_token_on():
    x, y = batch(torch.arange(1000), 64, 8, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 64) and (y == x + 1).all()


def test_a_few_steps_lower_the_loss():
    torch.manual_seed(0)
    model, generator = small(), torch.Generator().manual_seed(0)
    ids = torch.randint(5, (2000,))
    first = evaluate(model, ids)
    for _ in range(20):
        loss = loss_on(model, *batch(ids, 64, 8, generator))
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= 0.1 * p.grad
    assert evaluate(model, ids) < first - 0.5


def test_sampling_repeats_with_the_same_seed():
    torch.manual_seed(0)
    model, start = small(), torch.tensor([1, 2, 3])
    a = sample(model, start, 10, torch.Generator().manual_seed(0))
    assert a == sample(model, start, 10, torch.Generator().manual_seed(0)) and len(a) == 13


def test_we_can_read_the_weight_file_format(tmp_path):
    numbers = torch.arange(6, dtype=torch.float32).view(2, 3)
    header = json.dumps({"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}).encode()
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + numbers.numpy().tobytes())
    assert torch.equal(read_safetensors(path)["w"], numbers)


@pytest.mark.reference
@pytest.mark.skipif(not WEIGHTS.exists(), reason="download GPT-2's weights first; see lesson 10")
def test_our_model_runs_gpt2s_weights_and_gives_gpt2s_scores():
    reference = json.loads(Path("tests/gpt2_reference.json").read_text())
    with torch.no_grad():
        scores = load_gpt2(WEIGHTS)(torch.tensor([reference["ids"]]))[0]
    assert scores.argmax(dim=-1).tolist() == reference["argmax"]
    ours = scores[-1, reference["check"]]
    assert torch.allclose(ours, torch.tensor(reference["last"]), atol=1e-4)
