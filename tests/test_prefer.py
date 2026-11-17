import math

import pytest
import torch

from gpt.model import GPT
from gpt.prefer import preference_loss
from gpt.reinforce import answer_log_chances


def test_preference_loss_starts_at_ln_2_and_falls_as_the_better_answer_gains():
    same = torch.tensor([-3.0, -5.0])
    assert preference_loss(same, same - 1, same, same - 1).item() == pytest.approx(math.log(2))
    ahead = preference_loss(same + 2, same - 1, same, same - 1).item()
    behind = preference_loss(same - 2, same - 1, same, same - 1).item()
    assert ahead < math.log(2) < behind


def test_preference_loss_by_hand():
    # We favour the better answer by 4 more than the frozen model does. beta 0.1: -ln sigmoid(0.4)
    loss = preference_loss(
        torch.tensor([-1.0]), torch.tensor([-6.0]), torch.tensor([-2.0]), torch.tensor([-3.0])
    )
    assert loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-0.4))))


def test_a_step_makes_the_better_answer_gain_on_the_worse_one():
    torch.manual_seed(0)
    net = GPT(vocab=11, context=16, width=32, heads=2, layers=2)
    better, worse = [([1, 2, 9, 3, 10], 3)], [([1, 2, 9, 4, 10], 3)]  # one prompt, two answers
    with torch.no_grad():  # the reference: the model before it moves
        fixed = [answer_log_chances(net, part, pad=10) for part in (better, worse)]
    ours = [answer_log_chances(net, part, pad=10) for part in (better, worse)]
    loss = preference_loss(*ours, *fixed)
    assert loss.item() == pytest.approx(math.log(2))  # our model is the reference: an even bet
    loss.backward()
    with torch.no_grad():
        for p in net.parameters():
            p -= 0.05 * p.grad
        now = [answer_log_chances(net, part, pad=10) for part in (better, worse)]
    assert (now[0] - fixed[0]).item() > (now[1] - fixed[1]).item()  # the lead is above 0
