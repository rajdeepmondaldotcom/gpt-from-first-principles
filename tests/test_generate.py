import pytest
import torch

from gpt.generate import choose, generate
from gpt.model import GPT

SCORES = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()  # chances of 0.5, 0.3, 0.15 and 0.05


def share(**how):
    """How often each token is chosen in 20,000 draws."""
    generator = torch.Generator().manual_seed(0)
    picks = torch.tensor([choose(SCORES, generator=generator, **how) for _ in range(20000)])
    return picks.bincount(minlength=4) / 20000


def model():
    torch.manual_seed(0)
    return GPT(vocab=65, context=32, width=32, heads=4, layers=2, kv_heads=2)


def test_temperature_0_is_the_top_score_every_time():
    assert {choose(SCORES, temperature=0) for _ in range(20)} == {0}
    assert choose(SCORES.flip(0), temperature=0) == 3  # the top score, wherever it sits


def test_at_temperature_0_the_text_is_the_top_token_at_each_step():
    net, ids = model(), [5, 9, 2]
    for _ in range(10):
        ids.append(net(torch.tensor([ids]))[0, -1].argmax().item())
    assert generate(net, [5, 9, 2], 10, temperature=0) == ids


def test_at_temperature_1_each_token_comes_as_often_as_its_chance():
    assert torch.allclose(share(), torch.tensor([0.5, 0.3, 0.15, 0.05]), atol=0.01)


def test_temperature_by_hand():
    # Halving the temperature squares each chance: 0.25, 0.09, 0.0225, 0.0025, which sum to 0.365.
    assert torch.allclose(
        share(temperature=0.5), torch.tensor([0.685, 0.247, 0.062, 0.007]), atol=0.01
    )
    hot = share(temperature=100.0)
    assert torch.allclose(hot, torch.full((4,), 0.25), atol=0.02)  # nearly even


@pytest.mark.parametrize(
    ("top_p", "kept"), [(0.0, 1), (0.5, 1), (0.6, 2), (0.79, 2), (0.81, 3), (1.0, 4)]
)
def test_top_p_keeps_the_smallest_set_that_reaches_p(top_p, kept):
    assert (share(top_p=top_p) > 0).sum().item() == kept  # the likeliest token always stays


def test_top_p_shares_the_chance_among_the_tokens_it_keeps():
    assert torch.allclose(share(top_p=0.75), torch.tensor([0.625, 0.375, 0, 0]), atol=0.01)


def test_choosing_does_not_change_the_scores():
    before = SCORES.clone()
    choose(SCORES, top_p=0.5)
    assert torch.equal(SCORES, before)


def test_with_and_without_a_cache_the_text_is_the_same():
    net, prompt = model(), [5, 9, 2]
    cached = generate(net, prompt, 20, temperature=0)
    assert cached == generate(net, prompt, 20, temperature=0, cache=False)
    assert cached[:3] == prompt and len(cached) == 23


def test_with_a_cache_the_model_reads_each_token_once():
    net, read = model(), []
    net.embed.register_forward_hook(lambda layer, inputs, out: read.append(inputs[0].numel()))
    generate(net, [5, 9, 2], 20, temperature=0)
    assert sum(read) == 3 + 19  # the prompt, then each new token but the last
    read.clear()
    generate(net, [5, 9, 2], 20, temperature=0, cache=False)
    assert sum(read) == sum(range(3, 23))  # the whole text again for each new token: 250


def test_it_stops_at_the_context_the_model_trained_on():
    net = model()  # a context of 32
    for cache in (True, False):
        assert len(generate(net, [5, 9, 2], 100, temperature=0, cache=cache)) == 32


def test_it_stops_at_the_stop_token():
    net = model()
    free = generate(net, [5, 9, 2], 20, temperature=0)
    stopped = generate(net, [5, 9, 2], 20, temperature=0, stop=free[6])
    assert stopped == free[: free.index(free[6], 3) + 1]
