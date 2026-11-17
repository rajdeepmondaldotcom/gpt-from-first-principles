import math
from pathlib import Path

import torch
import torch.nn.functional as F

from gpt.autograd import learn_bigram as learn_one_number_at_a_time
from gpt.tensors import cross_entropy, cross_entropy_slopes, encode, learn_bigram, sample, softmax

SENTENCE = "the cat sat on the mat"


def test_hand_example():
    scores, target = torch.tensor([[2.0, 1.0, 0.0]]), torch.tensor([0])
    assert torch.allclose(softmax(scores), torch.tensor([[0.665, 0.245, 0.090]]), atol=5e-4)
    assert math.isclose(cross_entropy(scores, target), 0.408, abs_tol=5e-4)
    slopes = cross_entropy_slopes(scores, target)
    assert torch.allclose(slopes, torch.tensor([[-0.335, 0.245, 0.090]]), atol=5e-4)
    assert math.isclose(slopes.sum(), 0, abs_tol=1e-6)


def test_ours_equal_torch():
    torch.manual_seed(0)
    scores, targets = torch.randn(50, 7, dtype=torch.float64), torch.randint(7, (50,))
    assert torch.allclose(softmax(scores), torch.softmax(scores, dim=-1))
    assert torch.allclose(cross_entropy(scores, targets), F.cross_entropy(scores, targets))


def test_slopes_by_hand_equal_autograd():
    torch.manual_seed(0)
    scores = torch.randn(50, 7, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(7, (50,))
    cross_entropy(scores, targets).backward()
    assert torch.allclose(cross_entropy_slopes(scores.detach(), targets), scores.grad)


def test_big_scores_do_not_overflow():
    big = torch.tensor([[1000.0, 999.0, 998.0]])
    naive = big.exp() / big.exp().sum()
    assert naive.isnan().all()  # e to the 1000 is too big for the machine: inf / inf
    assert torch.allclose(softmax(big), softmax(torch.tensor([[2.0, 1.0, 0.0]])))
    assert math.isclose(cross_entropy(big * 10, torch.tensor([0])), 0, abs_tol=1e-4)
    far = torch.tensor([[0.0, -200.0]])  # the target's chance is e to the -200: 0 in 32 bits
    assert (-(far.exp() / far.exp().sum()).log())[0, 1] == math.inf
    assert cross_entropy(far, torch.tensor([1])) == 200


def test_a_lookup_sends_its_slopes_back_to_the_row_it_read():
    torch.manual_seed(0)
    W = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    x, y = torch.tensor([0, 3, 3, 1]), torch.tensor([1, 2, 4, 0])
    cross_entropy(W[x], y).backward()
    by_hand = torch.zeros_like(W).index_add_(0, x, cross_entropy_slopes(W[x].detach(), y))
    assert torch.allclose(by_hand, W.grad)
    assert W.grad[3].abs().sum() > 0 and W.grad[2].abs().sum() == 0  # row 2 was never read


def test_a_sample_follows_the_row_of_its_last_id():
    W = torch.full((3, 3), -1e9)
    W[0, 1] = W[1, 2] = W[2, 0] = 0.0  # 0 is followed by 1, 1 by 2 and 2 by 0, each time
    assert sample(W, torch.tensor([0]), 5, torch.Generator().manual_seed(0)) == [0, 1, 2, 0, 1, 2]


def test_same_numbers_as_lesson_2():
    vocab = sorted(set(SENTENCE))
    ids = encode(SENTENCE, vocab)
    before = learn_one_number_at_a_time(SENTENCE, steps=30)
    for steps in (1, 10, 30):
        W = learn_bigram(ids, len(vocab), steps=steps, rate=5.0)
        after = learn_one_number_at_a_time(SENTENCE, steps=steps + 1)[-1]
        assert math.isclose(cross_entropy(W[ids[:-1]], ids[1:]), after, rel_tol=1e-5)
    assert math.isclose(before[0], math.log(10))


def test_random_batches_get_close_to_the_whole_text():
    text = Path("data/tiny_shakespeare.txt").read_text()[:20000]
    vocab = sorted(set(text))
    ids = encode(text, vocab)

    def loss(W):
        return cross_entropy(W[ids[:-1]], ids[1:]).item()

    whole = learn_bigram(ids, len(vocab), steps=200, rate=30.0)
    batches = learn_bigram(
        ids,
        len(vocab),
        steps=200,
        rate=30.0,
        batch=1000,
        generator=torch.Generator().manual_seed(0),
    )
    assert loss(torch.zeros(len(vocab), len(vocab))) > 4.0  # untrained: a blind guess
    assert loss(whole) < 2.6 and abs(loss(whole) - loss(batches)) < 0.05
