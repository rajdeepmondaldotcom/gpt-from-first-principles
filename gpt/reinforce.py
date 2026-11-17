"""Lessons 27 and 29. Learning from a score, and from which of two answers is better."""

import torch

from gpt.finetune import collate
from gpt.tensors import cross_entropy


def answer_log_chances(model, examples, pad):
    """For each example, the log of the chance the model gives its whole answer."""
    x, y, scored = collate(examples, pad, model.embed.weight.device)
    scores = model(x)
    each = -torch.stack(
        [cross_entropy(scores[row][scored[row]], y[row][scored[row]]) for row in range(len(x))]
    )
    return each * scored.sum(dim=1)  # cross_entropy gave a mean; the answer's log chance is a sum


def score_loss(log_chances, rewards, groups):
    """Raise the chance of answers that scored above their group's average, and lower the rest."""
    rewards = torch.as_tensor(rewards, dtype=log_chances.dtype, device=log_chances.device)
    groups = torch.as_tensor(groups, device=log_chances.device)
    baseline = torch.stack([rewards[groups == g].mean() for g in groups])
    return -((rewards - baseline) * log_chances).mean()


def preference_loss(ours_better, ours_worse, frozen_better, frozen_worse, beta=0.1):
    """DPO. How much more we favour the better answer than a frozen copy of the model does."""
    lead = (ours_better - frozen_better) - (ours_worse - frozen_worse)
    return -torch.nn.functional.logsigmoid(beta * lead).mean()
