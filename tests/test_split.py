"""Lesson 43. Shared out among workers, every sum must come out as it did in one piece."""

import pytest
import torch

from gpt.gpt2 import batch, loss_on
from gpt.model import GPT, SwiGLU
from gpt.optim import AdamW
from gpt.split import deal, dealt, held_bytes, part, shares


def test_the_shares_cover_each_thing_once_and_are_nearly_equal():
    for count, workers in ((10, 3), (128, 4), (5, 5), (7, 1)):
        cut = shares(count, workers)
        assert [i for s in cut for i in range(count)[s]] == list(range(count))
        sizes = [len(range(count)[s]) for s in cut]
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("workers", [1, 2, 3, 5])
def test_the_workers_parts_add_up_to_the_whole_mlp(workers):
    torch.manual_seed(0)
    mlp, x = SwiGLU(48), torch.randn(4, 6, 48)
    parts = [part(held, x) for held in deal(mlp, workers)]
    assert torch.allclose(sum(parts), mlp(x), atol=1e-5)
    held = sum(t.numel() for worker in deal(mlp, workers) for t in worker)
    assert held == sum(p.numel() for p in mlp.parameters())  # each weight is held, and once
    if workers > 1:  # one worker left out, and the sum is no longer the result
        assert not torch.allclose(sum(parts[:-1]), mlp(x), atol=1e-3)


def test_each_worker_finds_its_own_slopes_and_the_inputs_slope_is_their_sum():
    torch.manual_seed(0)
    mlp, workers = SwiGLU(48), 3
    x, up = torch.randn(4, 6, 48, requires_grad=True), torch.randn(4, 6, 48)
    (mlp(x) * up).sum().backward()  # the whole MLP, with `up` as the slope from the next layer
    back = torch.zeros_like(x)
    for units, mine in zip(shares(mlp.narrow.in_features, workers), deal(mlp, workers)):
        own = [t.detach().clone().requires_grad_(True) for t in mine]
        seen = x.detach().clone().requires_grad_(True)
        gate, wide, narrow, mine_back = torch.autograd.grad(
            part(own, seen), [*own, seen], grad_outputs=up
        )
        assert torch.allclose(gate, mlp.gate.weight.grad[units], atol=1e-5)
        assert torch.allclose(wide, mlp.wide.weight.grad[units], atol=1e-5)
        assert torch.allclose(narrow, mlp.narrow.weight.grad[:, units], atol=1e-5)
        back += mine_back
    assert torch.allclose(back, x.grad, atol=1e-5)
    assert not torch.allclose(mine_back, x.grad, atol=1e-3)  # one worker's part is not enough


def test_adamw_with_its_averages_shared_out_ends_where_one_adamw_does():
    def trained(optimizers):
        torch.manual_seed(0)
        model = GPT(vocab=32, context=16, width=48, heads=4, kv_heads=2, layers=2)
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(32, (500,), generator=generator)
        steppers = optimizers(list(model.parameters()))
        for _ in range(4):
            model.zero_grad()
            loss_on(model, *batch(ids, 16, 8, generator)).backward()
            for stepper in steppers:
                stepper.step()
        return model, steppers

    one, _ = trained(lambda params: [AdamW(params, rate=1e-2)])
    many, steppers = trained(lambda params: dealt(params, 3, rate=1e-2))
    assert all(torch.equal(a, b) for a, b in zip(one.parameters(), many.parameters()))
    kept = sum(m.numel() for stepper in steppers for m in stepper.mean)
    assert kept == sum(p.numel() for p in one.parameters())  # each average is kept, and once
    idle, _ = trained(lambda params: dealt(params, 3, rate=1e-2)[:2])  # a worker that never steps
    assert not all(torch.equal(a, b) for a, b in zip(one.parameters(), idle.parameters()))


def test_a_worker_holds_sixteen_bytes_a_parameter_or_fewer_with_the_averages_shared():
    model = GPT(vocab=32, context=16, width=48, heads=4, kv_heads=2, layers=2)
    count = sum(p.numel() for p in model.parameters())
    assert held_bytes(model, count) == 16 * count  # lesson 17's count
    assert held_bytes(model, count // 4) == 8 * count + 8 * (count // 4)
