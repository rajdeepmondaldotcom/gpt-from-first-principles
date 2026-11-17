import pytest
import torch

from gpt.model import GPT
from gpt.reinforce import answer_log_chances, sample, score_loss
from gpt.tasks import prompt, score

END, SEP = 10, 9


def model():
    torch.manual_seed(0)
    return GPT(vocab=11, context=16, width=32, heads=2, layers=2)


def test_an_answers_log_chance_is_the_sum_over_its_tokens():
    net = model()
    examples = [([1, 2, SEP, 3, 4, END], 3), ([5, SEP, 6, END], 2)]
    got = answer_log_chances(net, examples, pad=END)
    for (ids, start), mine in zip(examples, got):
        logs = torch.log_softmax(net(torch.tensor([ids[:-1]]))[0], dim=-1)
        by_hand = sum(logs[i - 1, ids[i]] for i in range(start, len(ids)))
        assert mine.item() == pytest.approx(by_hand.item(), abs=1e-5)


def exact_and_sampled(group, samples=200_000):
    """A policy over 5 answers with known rewards: the true slope of the mean reward, and estimates.

    Each sample's reward has the average of its group of `group` samples taken off. 0 takes nothing.
    """
    scores = torch.tensor([0.5, -0.2, 0.1, 0.9, -0.6], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    (torch.softmax(scores, 0) * rewards).sum().backward()  # the mean reward, exactly
    exact = scores.grad.clone()
    generator = torch.Generator().manual_seed(0)
    picks = torch.multinomial(torch.softmax(scores.detach(), 0), samples, True, generator=generator)
    onehot = torch.nn.functional.one_hot(picks, 5).float()
    slopes = onehot - torch.softmax(scores.detach(), 0)  # the slope of each pick's log chance
    taken = rewards[picks]
    centre = taken.view(-1, group).mean(dim=1).repeat_interleave(group) if group else 0.0
    each = (taken - centre)[:, None] * slopes  # one estimate of the slope per sample
    return exact, each


def test_the_sampled_slope_averages_to_the_exact_one():
    exact, each = exact_and_sampled(group=0)
    assert torch.allclose(each.mean(dim=0), exact, atol=0.005)


def test_a_baseline_keeps_the_average_and_cuts_the_spread():
    exact, plain = exact_and_sampled(group=0)
    _, centred = exact_and_sampled(group=200_000)  # one group: the average of every sample
    assert torch.allclose(centred.mean(dim=0), exact, atol=0.005)
    assert centred.var(dim=0).sum() < 0.6 * plain.var(dim=0).sum()


def test_a_group_of_four_keeps_the_direction_and_three_quarters_of_the_size():
    # The average of 4 includes the sample itself: r - mean of 4 = 3/4 × (r - mean of the other 3).
    exact, plain = exact_and_sampled(group=0)
    _, grouped = exact_and_sampled(group=4)
    assert torch.allclose(grouped.mean(dim=0), 0.75 * exact, atol=0.005)
    assert (grouped / 0.75).var(dim=0).sum() < 0.8 * plain.var(dim=0).sum()


def test_score_loss_follows_the_slope_of_the_mean_reward():
    # Two prompts, two sampled answers each. The loss's slope for each log chance is minus
    # (reward - its group's mean) / 4.
    logs = torch.zeros(4, requires_grad=True)
    score_loss(logs, rewards=[1.0, 0.0, 1.0, 1.0], groups=[0, 0, 1, 1]).backward()
    assert torch.allclose(logs.grad, torch.tensor([-0.125, 0.125, 0.0, 0.0]))


def test_a_step_on_score_loss_raises_the_chance_of_the_answer_that_passed():
    net = model()
    examples = [([1, 2, SEP, 3, END], 3), ([1, 2, SEP, 4, END], 3)]  # one prompt, two answers
    before = answer_log_chances(net, examples, pad=END).detach()
    loss = score_loss(answer_log_chances(net, examples, pad=END), [1.0, 0.0], [0, 0])
    loss.backward()
    with torch.no_grad():
        for p in net.parameters():
            p -= 0.05 * p.grad
    after = answer_log_chances(net, examples, pad=END).detach()
    assert after[0] > before[0] and after[1] < before[1]


class Bytes:
    """A tokenizer of single bytes, for a test. 256 ends a story."""

    end = 256

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode(errors="replace")


def test_a_sampled_story_comes_back_as_an_example_with_its_score():
    torch.manual_seed(0)
    net = GPT(vocab=257, context=64, width=32, heads=2, layers=2)
    task = {"name": "Mira", "words": ["cat", "moon", "brave"]}
    asked = Bytes().encode(prompt(task))
    (ids, start), mark = sample(net, Bytes(), task, torch.Generator().manual_seed(0))
    assert ids[:start] == asked and start == len(asked)  # the prompt, then what the model wrote
    assert len(ids) == 64 or ids[-1] == 256  # it wrote to the end of the context, or stopped
    story = Bytes().decode([i for i in ids[start:] if i != 256])
    assert mark == score(task, story)  # scored on the story alone, never on the prompt
    again, _ = sample(net, Bytes(), task, torch.Generator().manual_seed(1))
    assert again[0] != ids  # drawn by chance: another seed, another story
