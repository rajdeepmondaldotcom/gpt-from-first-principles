import torch

from gpt.cost import cache_bytes
from gpt.engine import Engine, Request, slide
from gpt.generate import generate
from gpt.model import GPT

PROMPTS = [[5, 9, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9], list(range(10, 27)), [40], [7, 7, 7, 7, 7, 7]]
LIMITS = [5, 12, 8, 20, 3]


def model():
    torch.manual_seed(0)
    net = GPT(vocab=65, context=64, width=32, heads=4, layers=2, kv_heads=2)
    with torch.no_grad():
        net.embed.weight *= 5  # what it writes now depends on each token before, and on where
    return net


def alone(net, prompt, limit, stop=None):
    return generate(net, prompt, limit, stop=stop, temperature=0)[len(prompt) :]


def serve(net, arrivals, chunk=64, budget=10**9):
    """Run an engine. `arrivals` maps a round number to the requests that arrive in it."""
    engine, rounds = Engine(net, budget, chunk), 0
    while engine.reading or engine.rows or any(r >= rounds for r in arrivals):
        for request in arrivals.get(rounds, []):
            assert engine.admit(request)
        engine.step()
        rounds += 1
    return rounds


def test_each_row_of_a_batch_matches_its_solo_run():
    net = model()
    requests = [Request(p, n, how={"temperature": 0}) for p, n in zip(PROMPTS, LIMITS)]
    serve(net, {0: requests})
    for request in requests:
        assert request.written == alone(net, request.prompt, request.limit)


def test_requests_join_and_leave_a_running_batch_and_nothing_changes():
    net = model()
    requests = [Request(p, n, how={"temperature": 0}) for p, n in zip(PROMPTS, LIMITS)]
    serve(net, {0: requests[:1], 2: requests[1:3], 7: requests[3:4], 9: requests[4:]})
    for request in requests:
        assert request.written == alone(net, request.prompt, request.limit)


def test_each_row_draws_with_its_own_settings():
    net = model()
    greedy = Request(PROMPTS[0], 9, how={"temperature": 0})
    chancy = Request(
        PROMPTS[1], 9, how={"temperature": 1.5, "generator": torch.Generator().manual_seed(3)}
    )
    serve(net, {0: [greedy, chancy]})
    assert greedy.written == alone(net, PROMPTS[0], 9)
    drawn = generate(
        net, PROMPTS[1], 9, temperature=1.5, generator=torch.Generator().manual_seed(3)
    )
    assert chancy.written == drawn[len(PROMPTS[1]) :] != alone(net, PROMPTS[1], 9)


def test_a_stop_token_ends_one_row_and_not_the_others():
    net = model()
    stop = alone(net, PROMPTS[1], 12)[4]
    requests = [
        Request(PROMPTS[1], 12, stop=stop, how={"temperature": 0}),
        Request(PROMPTS[0], 9, how={"temperature": 0}),
    ]
    serve(net, {0: requests})
    assert (
        requests[0].written == alone(net, PROMPTS[1], 12, stop=stop)
        and len(requests[0].written) < 12
    )
    assert requests[1].written == alone(net, PROMPTS[0], 9)


def test_a_long_prompt_is_read_in_chunks_and_the_answer_is_the_same():
    net = model()
    long, short = (
        Request(PROMPTS[2], 8, how={"temperature": 0}),
        Request(PROMPTS[0], 5, how={"temperature": 0}),
    )
    engine = Engine(net, 10**9, chunk=4)
    assert engine.admit(long) and engine.admit(short)
    engine.step()
    assert len(short.written) == 1 and not long.written  # the short one is not kept waiting
    while engine.reading or engine.rows:
        engine.step()
    assert long.written == alone(net, PROMPTS[2], 8) and short.written == alone(net, PROMPTS[0], 5)


def test_one_pass_writes_a_token_for_every_row():
    net, calls = model(), []
    net.embed.register_forward_hook(lambda part, inputs, out: calls.append(inputs[0].shape))
    engine = Engine(net, 10**9)
    for prompt in PROMPTS[:4]:
        engine.admit(Request(prompt, 10, how={"temperature": 0}))
    engine.step()  # each reads its prompt alone
    calls.clear()
    engine.step()
    assert [tuple(shape) for shape in calls] == [(4, 1)]  # four rows, one new token each, one pass


def test_the_budget_refuses_what_does_not_fit_and_room_comes_back():
    net = model()
    each = cache_bytes(net, 3 + 5)
    engine = Engine(net, budget=2 * each)
    first, second, third = (Request(PROMPTS[0], 5, how={"temperature": 0}) for _ in range(3))
    assert engine.admit(first) and engine.admit(second) and not engine.admit(third)
    while engine.reading or engine.rows:
        engine.step()
    assert engine.admit(third)


def test_the_cache_stays_inside_the_budget_when_rows_are_padded():
    net = model()
    long, short = list(range(1, 41)), [5, 9]
    budget = cache_bytes(net, 45) + 4 * cache_bytes(net, 7)  # what the five would need unpadded
    engine, peak = Engine(net, budget), 0
    asked = [Request(long, 5, how={"temperature": 0})]
    asked += [Request(short, 5, how={"temperature": 0}) for _ in range(4)]
    taken = [engine.admit(request) for request in asked]
    assert taken == [True, False, False, False, False]  # beside the long one, each counts as long
    assert Engine(net, 5 * cache_bytes(net, 45)).admit(asked[0])
    while engine.reading or engine.rows:
        engine.step()
        if engine.kept:
            peak = max(peak, sum(t.nbytes for one in engine.kept for t in (one["k"], one["v"])))
    assert 0 < peak <= budget


def test_sliding_a_row_along_the_batch_changes_no_score():
    net, ids = model(), torch.tensor([[5, 9, 2, 11, 30]])
    kept = [{} for _ in net.blocks]
    with torch.no_grad():
        net(ids, kept)
        plain = net(torch.tensor([[7]]), [dict(one) for one in kept])
        slide(kept, 6, 6)  # six empty places in front, keys turned six places on
        real = torch.tensor([[False] * 6 + [True] * 6])
        for one in kept:
            one["real"] = real
        moved = net(torch.tensor([[7]]), kept)
    assert torch.allclose(plain, moved, atol=1e-5)


def test_without_the_mask_the_padding_is_read_and_the_scores_change():
    net, ids = model(), torch.tensor([[5, 9, 2, 11, 30]])
    kept = [{} for _ in net.blocks]
    with torch.no_grad():
        net(ids, kept)
        plain = net(torch.tensor([[7]]), [dict(one) for one in kept])
        slide(kept, 6, 6)
        unmasked = net(torch.tensor([[7]]), kept)
    assert not torch.allclose(plain, unmasked, atol=1e-3)
