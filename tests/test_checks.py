"""Each planted bug must be caught by the check named for it, and by a healthy model's passing."""

import math

import torch

from gpt import attention
from gpt.checks import (
    add_dropout,
    every_part_moves_a_little,
    learns_one_batch,
    reads_no_future,
    starts_with_no_opinion,
    step_sizes,
    targets_are_one_token_on,
)
from gpt.gpt2 import GPT2, batch, loss_on
from gpt.optim import AdamW

IDS = (torch.arange(2000) * 3) % 11


def healthy():
    torch.manual_seed(0)
    model = GPT2(vocab=11, context=16, width=32, heads=2, layers=2)
    x, y = batch(IDS, 16, 8, torch.Generator().manual_seed(0))
    return model, x, y


def test_a_healthy_model_passes_every_check():
    model, x, y = healthy()
    assert starts_with_no_opinion(model, IDS, vocab=11)
    assert targets_are_one_token_on(x, y)
    assert reads_no_future(model, x)
    # 3e-5 is lesson 12's peak rate on the first step of its warmup; 3e-3 below is 100 times that.
    assert every_part_moves_a_little(step_sizes(model, AdamW(model.parameters(), rate=3e-5), x, y))
    assert learns_one_batch(model, AdamW(model.parameters(), rate=1e-2), x, y)


def test_bug_starting_weights_twenty_times_too_large():
    model, _, _ = healthy()
    with torch.no_grad():
        model.embed.token.weight *= 20
    assert not starts_with_no_opinion(model, IDS, vocab=11)


def test_bug_targets_not_moved_on():
    _, x, _ = healthy()
    assert not targets_are_one_token_on(x, x)  # y = x: the model would learn to copy its input


def test_bug_no_mask(monkeypatch):
    model, x, _ = healthy()
    unmasked = attention.attend
    monkeypatch.setattr(attention, "attend", lambda q, k, v, mask=None: unmasked(q, k, v))
    assert not reads_no_future(model, x)


def test_a_model_with_no_mask_trains_without_complaint(monkeypatch):
    """The reason for this lesson: the bug prints a falling loss, lower than a healthy model's."""
    ids = torch.randint(11, (2000,), generator=torch.Generator().manual_seed(0))  # nothing to learn

    def loss_after_100_steps():
        torch.manual_seed(0)
        model = GPT2(vocab=11, context=16, width=32, heads=2, layers=2)
        optimizer, generator = (
            AdamW(model.parameters(), rate=1e-2),
            torch.Generator().manual_seed(0),
        )
        for _ in range(100):
            loss = loss_on(model, *batch(ids, 16, 8, generator))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item()

    healthy_loss = loss_after_100_steps()
    unmasked = attention.attend
    monkeypatch.setattr(attention, "attend", lambda q, k, v, mask=None: unmasked(q, k, v))
    assert healthy_loss > 2.0  # random text: close to ln 11 = 2.40
    assert loss_after_100_steps() < healthy_loss - 1.0  # it reads each answer from the future


def test_bug_rate_a_hundred_times_too_high():
    model, x, y = healthy()
    assert not every_part_moves_a_little(
        step_sizes(model, AdamW(model.parameters(), rate=3e-3), x, y)
    )


def test_bug_part_of_the_model_left_out_of_the_optimizer():
    model, x, y = healthy()
    sizes = step_sizes(model, AdamW(model.blocks.parameters(), rate=3e-5), x, y)
    assert sizes["embed.position.weight"] == 0 and not every_part_moves_a_little(sizes)


def test_bug_slopes_never_cleared(monkeypatch):
    model, x, y = healthy()
    monkeypatch.setattr(model, "zero_grad", lambda: None)  # each step now adds to the last slopes
    assert not learns_one_batch(model, AdamW(model.parameters(), rate=1e-2), x, y)


def test_dropout_zeroes_while_training_and_does_nothing_after():
    model, x, _ = healthy()
    with torch.no_grad():
        plain = model(x)
        add_dropout(model, 0.5)
        assert not torch.allclose(model(x), plain)  # training mode: a random half is dropped
        assert not torch.equal(model(x), model(x))  # ... a different half each time
        model.eval()
        assert torch.equal(model(x), plain)  # switched off: the same model as before
    assert math.isclose(torch.nn.Dropout(0.5)(torch.ones(100_000)).mean(), 1.0, abs_tol=0.02)
