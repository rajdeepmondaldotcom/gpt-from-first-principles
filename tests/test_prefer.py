import math

import pytest
import torch

from gpt.prefer import preference_loss


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
