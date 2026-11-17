import itertools

import pytest
import torch

from gpt.model import GPT
from gpt.speculate import imitation_loss, keep_chance, leftover, speculate
from gpt.tensors import cross_entropy, softmax


def model(seed, width=32):
    torch.manual_seed(seed)
    net = GPT(vocab=5, context=16, width=width, heads=2, layers=2)
    with torch.no_grad():
        net.embed.weight *= 3  # chances far from even, and different in the two models
    return net


def test_keep_chance_and_leftover_by_hand():
    p, q = torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.2, 0.6, 0.2])
    assert torch.allclose(keep_chance(p, q), torch.tensor([1.0, 0.5, 1.0]))
    assert torch.allclose(leftover(p, q), torch.tensor([1.0, 0.0, 0.0]))  # only token 0 is short


def test_imitation_loss_by_hand():
    scores = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]])
    sure = torch.tensor([[0.0, 50.0, 0.0], [50.0, 0.0, 0.0]])  # a big model sure of tokens 1 and 0
    one_right_token = cross_entropy(scores, torch.tensor([1, 0]))
    assert imitation_loss(scores, sure).item() == pytest.approx(one_right_token.item(), abs=1e-5)
    even = torch.zeros(1, 4)  # a big model with even chances, and a draft that matches it
    assert imitation_loss(even, even).item() == pytest.approx(torch.tensor(4.0).log().item())
    off = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    assert imitation_loss(off, even) > imitation_loss(even, even)  # lowest when the two agree


def test_one_step_draws_from_the_big_models_chances_exactly():
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        p, q = (torch.rand(8, generator=generator) for _ in range(2))
        p, q = p / p.sum(), q / q.sum()
        kept = q * keep_chance(p, q)  # the draft picks a token, and it is kept
        out = kept + (1 - kept.sum()) * leftover(p, q)  # or it is rejected and drawn again
        assert torch.allclose(out, p, atol=1e-6)


def test_two_tokens_come_as_often_as_the_big_model_alone_would_write_them():
    big, draft, prompt = model(0), model(1, width=16), [1, 2]
    exact = {}
    with torch.no_grad():
        first = softmax(big(torch.tensor([prompt]))[0, -1])
        for a in range(5):
            second = softmax(big(torch.tensor([[*prompt, a]]))[0, -1])
            for b in range(5):
                exact[a, b] = (first[a] * second[b]).item()
    generator, seen, draws = torch.Generator().manual_seed(0), dict.fromkeys(exact, 0), 3000
    for _ in range(draws):
        ids, _ = speculate(big, draft, prompt, 2, k=2, generator=generator)
        seen[ids[2], ids[3]] += 1
    worst = max(
        abs(seen[pair] / draws - exact[pair]) for pair in itertools.product(range(5), repeat=2)
    )
    assert worst < 0.03  # 3,000 draws: a cell's share is good to about 0.01
    assert max(exact.values()) < 0.6  # the answer is not one sure thing


def test_a_draft_equal_to_the_big_model_is_always_kept():
    big, generator = model(0), torch.Generator().manual_seed(0)
    ids, counts = speculate(big, big, [1, 2], 20, k=4, generator=generator)
    assert counts["kept"] == counts["checked"] and len(ids) == 22
    assert counts["passes"] == 4  # 5 tokens a pass: 4 guesses and the free one


def test_a_poor_draft_costs_passes_and_not_correctness():
    big, draft, generator = model(0), model(1, width=16), torch.Generator().manual_seed(0)
    ids, counts = speculate(big, draft, [1, 2], 40, k=4, generator=generator)
    assert len(ids) == 42 and 0 < counts["kept"] < counts["checked"]
    # A pass always writes one token, and one more for each guess kept: the count has to balance.
    assert counts["passes"] + counts["kept"] >= 40 > counts["passes"] > 8
