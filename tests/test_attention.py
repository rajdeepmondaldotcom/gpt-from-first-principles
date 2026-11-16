import math

import torch
import torch.nn.functional as F

from gpt.attention import CausalSelfAttention, attend

# Lesson 6's hand example: three tokens, and the query that belongs to "sat".
KEYS = torch.tensor([[1.0, 0, 1, 0], [0, 2, 0, 2], [1, 1, 0, 0]])  # the, cat, sat
VALUES = torch.tensor([[1.0, 0], [0, 1], [1, 1]])
QUERY = torch.tensor([[0.0, 2, 0, 2]])


def test_hand_example():
    out, weights = attend(QUERY, KEYS, VALUES)
    assert (QUERY @ KEYS.T / 2).tolist() == [[0.0, 4.0, 1.0]]
    assert torch.allclose(weights, torch.tensor([[0.017, 0.936, 0.047]]), atol=5e-4)
    assert torch.allclose(out, torch.tensor([[0.064, 0.983]]), atol=5e-4)


def test_the_hand_example_is_one_row_of_the_whole_grid():
    queries = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1, 1], [0, 2, 0, 2]])  # one per token
    out, weights = attend(queries, KEYS, VALUES)
    assert weights.shape == (3, 3) and out.shape == (3, 2)
    assert torch.allclose(out[2], attend(QUERY, KEYS, VALUES)[0][0])
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))


def test_ours_equals_torch():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 5, 8, dtype=torch.float64) for _ in range(3))
    past = torch.ones(5, 5, dtype=torch.bool).tril()
    assert torch.allclose(attend(q, k, v)[0], F.scaled_dot_product_attention(q, k, v))
    assert torch.allclose(
        attend(q, k, v, past)[0], F.scaled_dot_product_attention(q, k, v, is_causal=True)
    )


def test_dividing_by_the_root_of_the_width_keeps_the_weights_spread():
    torch.manual_seed(0)
    q, k = torch.randn(100, 256), torch.randn(100, 256)
    raw = q @ k.T
    assert 15 < raw.std() < 17 and 0.9 < (raw / 16).std() < 1.1  # spread grows as the root of 256
    largest = lambda scores: scores.softmax(dim=-1).max(dim=-1).values.mean()
    assert largest(raw) > 0.8  # undivided, a token puts nearly all its weight on one other token
    assert largest(raw / 16) < 0.1 and torch.allclose(attend(q, k, k)[1], (raw / 16).softmax(-1))


def test_no_slots_shuffle_the_tokens_and_the_results_shuffle_with_them():
    torch.manual_seed(0)
    x, order = torch.randn(6, 8), torch.randperm(6)
    out = attend(x, x, x)[0]
    assert torch.allclose(attend(x[order], x[order], x[order])[0], out[order], atol=1e-6)


def test_the_same_weights_serve_any_length():
    layer = CausalSelfAttention(width=16, heads=4)
    assert layer(torch.randn(1, 3, 16)).shape == (1, 3, 16)
    assert layer(torch.randn(1, 30, 16)).shape == (1, 30, 16)


def test_a_token_cannot_read_the_future():
    torch.manual_seed(0)
    layer, x = CausalSelfAttention(width=16, heads=4), torch.randn(1, 6, 16)
    changed = x.clone()
    changed[0, 4] = torch.randn(16)  # replace token 4
    before, after = layer(x), layer(changed)
    assert torch.equal(before[0, :4], after[0, :4])  # tokens 0 to 3 did not move at all
    assert not torch.allclose(before[0, 4:], after[0, 4:])


def test_the_mask_gives_future_tokens_a_weight_of_exactly_zero():
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    weights = attend(x, x, x, torch.ones(5, 5, dtype=torch.bool).tril())[1]
    assert (weights.triu(1) == 0).all() and torch.allclose(weights.sum(dim=-1), torch.ones(5))
    assert weights[0, 0] == 1  # the first token can read only itself


def test_heads_split_the_vector_and_each_runs_attention_on_its_slice():
    torch.manual_seed(0)
    layer, x = CausalSelfAttention(width=16, heads=4), torch.randn(2, 5, 16)
    q, k, v = layer.qkv(x).chunk(3, dim=-1)
    past = torch.ones(5, 5, dtype=torch.bool).tril()
    slices = [
        attend(q[..., s], k[..., s], v[..., s], past)[0]
        for s in (slice(i, i + 4) for i in range(0, 16, 4))
    ]
    assert torch.allclose(layer(x), layer.proj(torch.cat(slices, dim=-1)), atol=1e-6)


def test_more_heads_cost_no_extra_parameters():
    count = lambda heads: sum(p.numel() for p in CausalSelfAttention(64, heads).parameters())
    assert count(1) == count(8) == 4 * 64 * 64 + 4 * 64
    assert math.isclose(math.sqrt(4), 2)  # the hand example divides its scores by this


def test_masking_by_hand():
    """Lesson 7: `cat` scores (the, cat, sat) as (2, 1, 3) and may not read `sat`."""
    scores, may_read = torch.tensor([2.0, 1.0, 3.0]), torch.tensor([True, True, False])
    weights = scores.masked_fill(~may_read, -math.inf).softmax(dim=-1)
    assert torch.allclose(weights, torch.tensor([0.731, 0.269, 0.0]), atol=5e-4)
    late = scores.softmax(dim=-1) * may_read  # the obvious way: zero the weight afterwards...
    assert torch.allclose(weights, late / late.sum())  # ...then rescale. Same answer.
    assert torch.ones(3, 3, dtype=torch.bool).tril().int().tolist() == [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
