import torch

from gpt.gpt2 import batch, loss_on
from gpt.workers import finish, launch, share, start

SHAPE = {"vocab": 11, "context": 16, "width": 32, "heads": 2, "layers": 2}
IDS = (torch.arange(2000) * 7 + torch.arange(2000) // 3) % 11


def batches(size, steps=3):
    generator = torch.Generator().manual_seed(1)
    return [batch(IDS, 16, size, generator) for _ in range(steps)]


def alone(size):
    """One process, the whole batch at once."""
    model, optimizer = start(SHAPE, seed=0, rate=1e-2)
    for x, y in batches(size):
        model.zero_grad()
        loss_on(model, x, y).backward()
        optimizer.step()
    return model.state_dict()


def together(count, size):
    pipes = launch(count, SHAPE, seed=0, rate=1e-2)
    for x, y in batches(size):
        share(pipes, x, y)
    return finish(pipes)


def close(a, b):
    return all(torch.allclose(a[name], b[name], atol=1e-5) for name in a)


def test_four_workers_end_with_the_weights_of_one_process_on_the_whole_batch():
    single, four = alone(32), together(4, 32)
    assert all(close(single, weights) for weights in four)
    assert all(torch.equal(four[0][name], other[name]) for other in four[1:] for name in four[0])


def test_unequal_shares_are_weighted_by_their_size():
    assert close(alone(30), together(4, 30)[0])  # shares of 8, 8, 8 and 6 windows

    def plain_mean(pipes, x, y):  # the planted bug: every worker counts the same
        xs, ys = x.chunk(len(pipes)), y.chunk(len(pipes))
        for pipe, part, target in zip(pipes, xs, ys):
            pipe.send((part, target))
        slopes = [pipe.recv() for pipe in pipes]
        mean = [sum(same) / len(pipes) for same in zip(*slopes)]
        for pipe in pipes:
            pipe.send(mean)

    pipes = launch(4, SHAPE, seed=0, rate=1e-2)
    for x, y in batches(30):
        plain_mean(pipes, x, y)
    assert not close(alone(30), finish(pipes)[0])
