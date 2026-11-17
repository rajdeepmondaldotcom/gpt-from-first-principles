import math
from itertools import pairwise

import torch

from gpt.autograd import Value, learn_bigram
from gpt.bigram import count, loss, table

SENTENCE = "the cat sat on the mat"


def expression(a, b, c):
    """Uses every operation Value knows: add, subtract, multiply, divide, power, exp, log."""
    return ((a * b + c) ** 2 / (a.exp() + 1)).log() - b / c


def test_our_gradients_equal_torch():
    numbers = [2.0, -3.0, 10.0]
    ours = [Value(x) for x in numbers]
    expression(*ours).backward()
    theirs = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in numbers]
    expression(*theirs).backward()
    for mine, reference in zip(ours, theirs):
        assert math.isclose(mine.grad, reference.grad.item(), rel_tol=1e-12)


def test_the_chain_rule_agrees_with_nudging():
    numbers, h = [2.0, -3.0, 10.0], 1e-6
    values = [Value(x) for x in numbers]
    expression(*values).backward()
    for i, v in enumerate(values):
        nudged = [Value(x + h * (i == j)) for j, x in enumerate(numbers)]
        slope = (expression(*nudged).data - expression(*values).data) / h
        assert math.isclose(v.grad, slope, rel_tol=1e-4)


def test_hand_example():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    L = a * b + c
    L.backward()
    assert (L.data, a.grad, b.grad, c.grad) == (4.0, -3.0, 2.0, 1.0)


def test_each_operation_knows_its_own_slope():
    """The table in lesson 2: power, e to the x, and ln, each checked against its rule."""
    for build, x, rule in [
        (lambda v: v**3, 2.0, 3 * 2.0**2),
        (lambda v: v.exp(), 1.5, math.exp(1.5)),
        (lambda v: v.log(), 4.0, 1 / 4.0),
    ]:
        v = Value(x)
        build(v).backward()
        assert math.isclose(v.grad, rule)


def test_a_number_used_twice_collects_both_paths():
    a = Value(3.0)
    (a + a).backward()
    assert a.grad == 2.0
    a = Value(3.0)
    (a * a).backward()
    assert a.grad == 6.0


def walk(rate, steps=50):
    """Gradient descent on (w - 3) squared from w = 0, with our own slopes."""
    w, path = Value(0.0), []
    for _ in range(steps):
        ((w - 3) ** 2).backward()
        w.data -= rate * w.grad
        w.grad = 0.0
        path.append(round(w.data, 4))
    return path


def test_walking_downhill_by_hand():
    path = walk(rate=0.1)
    assert path[:5] == [0.6, 1.08, 1.464, 1.7712, 2.017]
    assert math.isclose(path[-1], 3.0, abs_tol=1e-3)


def test_a_step_that_is_too_big_runs_away():
    assert abs(walk(rate=1.1)[-1] - 3) > 1000


def test_the_learned_table_reaches_the_counted_one():
    counted = loss(table(count(SENTENCE), sorted(set(SENTENCE))), SENTENCE)
    losses = learn_bigram(SENTENCE)
    assert math.isclose(losses[0], math.log(10))  # all scores 0: a blind guess
    assert all(later < earlier for earlier, later in pairwise(losses))
    assert 0 < losses[-1] - counted < 0.01  # closes in, and never passes it
