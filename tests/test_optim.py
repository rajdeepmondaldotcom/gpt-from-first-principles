import torch
from torch import nn

from gpt.gpt2 import GPT2, batch, loss_on
from gpt.optim import SGD, AdamW


def twins():
    """Two copies of one small GPT, and a fixed stream of batches to train them on."""
    torch.manual_seed(0)
    ours = GPT2(vocab=11, context=8, width=16, heads=2, layers=2)
    theirs = GPT2(vocab=11, context=8, width=16, heads=2, layers=2)
    theirs.load_state_dict(ours.state_dict())
    ids, generator = torch.randint(11, (500,)), torch.Generator().manual_seed(0)
    return ours, theirs, [batch(ids, 8, 4, generator) for _ in range(20)]


def train_both(ours, theirs, mine, reference, batches):
    for x, y in batches:
        for model, optimizer in ((ours, mine), (theirs, reference)):
            model.zero_grad()
            loss_on(model, x, y).backward()
            optimizer.step()
        for a, b in zip(ours.parameters(), theirs.parameters()):
            assert torch.allclose(a, b, atol=1e-6)


def test_our_gradient_descent_equals_torch_step_for_step():
    ours, theirs, batches = twins()
    train_both(
        ours,
        theirs,
        SGD(ours.parameters(), 0.1),
        torch.optim.SGD(theirs.parameters(), 0.1),
        batches,
    )


def test_our_momentum_equals_torch_step_for_step():
    ours, theirs, batches = twins()
    reference = torch.optim.SGD(theirs.parameters(), 0.1, momentum=0.9)
    train_both(ours, theirs, SGD(ours.parameters(), 0.1, momentum=0.9), reference, batches)


def test_our_adamw_equals_torch_step_for_step():
    ours, theirs, batches = twins()
    grids = [p for p in theirs.parameters() if p.dim() > 1]
    rest = [p for p in theirs.parameters() if p.dim() <= 1]
    groups = [{"params": grids, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}]
    train_both(
        ours, theirs, AdamW(ours.parameters(), 1e-3), torch.optim.AdamW(groups, 1e-3), batches
    )


def test_our_adamw_equals_torch_with_other_settings():
    ours, theirs, batches = twins()
    grids = [p for p in theirs.parameters() if p.dim() > 1]
    rest = [p for p in theirs.parameters() if p.dim() <= 1]
    groups = [{"params": grids, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}]
    mine = AdamW(ours.parameters(), 3e-3, betas=(0.8, 0.95), decay=0.1, eps=1e-6)
    train_both(
        ours, theirs, mine, torch.optim.AdamW(groups, 3e-3, betas=(0.8, 0.95), eps=1e-6), batches
    )


def one_number(slope):
    p = nn.Parameter(torch.tensor([5.0]))
    p.grad = torch.tensor([slope])
    return p


def test_momentum_by_hand():
    p = one_number(1.0)
    optimizer, places = SGD([p], rate=0.1, momentum=0.9), []
    for slope in (1.0, 1.0, -1.0):
        p.grad = torch.tensor([slope])
        optimizer.step()
        places.append(round(p.item(), 3))
    assert places == [
        4.9,
        4.71,
        4.639,
    ]  # steps of 0.1, 0.19 and 0.071: it keeps going after the turn


def test_adams_first_step_is_the_rate_whatever_the_slope():
    for slope in (0.001, 1.0, 1000.0):
        p = one_number(slope)
        AdamW([p], rate=0.01).step()
        assert torch.allclose(p, torch.tensor([4.99]), atol=1e-6)


def test_a_steady_slope_builds_momentum_to_ten_times_itself():
    p = one_number(1.0)
    optimizer = SGD([p], rate=0.0, momentum=0.9)
    for _ in range(200):
        optimizer.step()
    assert torch.allclose(optimizer.velocity[0], torch.tensor([10.0]), atol=1e-4)


def test_adam_moves_a_flipping_slope_far_less_than_a_steady_one():
    def travelled(slopes):
        p = one_number(0.0)
        optimizer = AdamW([p], rate=0.01)
        for slope in slopes:
            p.grad = torch.tensor([slope])
            optimizer.step()
        return abs(p.item() - 5.0)

    steady, flipping = travelled([0.5] * 20), travelled([0.5, -0.5] * 10)
    assert abs(steady - 0.2) < 1e-3  # 20 steps of the rate, 0.01 each
    assert flipping < steady / 5


def test_decay_shrinks_weight_grids_and_leaves_the_rest():
    grid, bias = nn.Parameter(torch.ones(2, 2)), nn.Parameter(torch.ones(2))
    grid.grad, bias.grad = torch.zeros(2, 2), torch.zeros(2)
    AdamW([grid, bias], rate=0.1, decay=0.5).step()
    assert torch.allclose(grid, torch.full((2, 2), 0.95)) and torch.equal(bias, torch.ones(2))


def test_one_rate_cannot_suit_slopes_that_differ_four_hundred_times():
    from pathlib import Path

    from gpt.tensors import encode

    text = Path("data/tiny_shakespeare.txt").read_text()
    ids = encode(text, sorted(set(text)))
    torch.manual_seed(0)
    model = GPT2(vocab=65, context=64, width=128, heads=4, layers=4)  # lesson 10's model, untrained
    loss_on(model, *batch(ids, 64, 32, torch.Generator().manual_seed(0))).backward()
    typical = {name: p.grad.abs().mean().item() for name, p in model.named_parameters()}
    grids = [typical[name] for name, p in model.named_parameters() if p.dim() > 1]
    assert 10 < max(grids) / min(grids) < 20  # between two grids of weights
    assert 300 < max(typical.values()) / min(typical.values()) < 500  # a bias against a gain
