# 20. Many workers

By the end of this page four processes train one model between them, and they end with the weights one process would have reached alone.

The run we picked is 8,921 steps on one graphics chip. A run on text the size of the web is too long for one machine. To use many machines, a step has to be split among them without changing its result.

## The idea

A loss is an average over the windows in a batch, and slopes add, as lesson 2 showed. So a batch can be cut into shares. Give each share to a worker that holds the same weights. Each worker finds the slopes for its own share. The average of those slopes, with each worker counted by the size of its share, is the slope of the whole batch.

Each worker is a separate process with its own copy of the model and its own optimizer. All of them start from one seed, so their weights begin equal. A step goes like this:

1. A coordinator draws the batch and sends each worker its share.
2. Each worker runs its share forward and back, and sends its slopes to the coordinator.
3. The coordinator averages the slopes and sends the average to each worker.
4. Each worker takes one optimizer step with that average.

Equal weights, the same slopes and the same optimizer give equal weights after the step. The weights themselves never travel.

The slopes do, and they are as large as the model: 4 bytes for each parameter, from each worker and back, on each step. Sending them takes time that one process alone would not spend. That is one reason four workers are less than four times as fast as one.

The other reason is hardware. Workers pay off when each has a processor or a graphics chip of its own. On one laptop they share the same cores, and the run below shows what that buys.

## By hand

Two workers. The first has 3 windows and finds an average slope of 2.0 for one parameter. The second has 1 window and finds 6.0.

| Way to combine | Sum | Slope |
|---|---|---|
| Counted by share | (3 × 2.0 + 1 × 6.0) / 4 | 3.0 |
| Plain mean of the two | (2.0 + 6.0) / 2 | 4.0 |

One process on all four windows would find 3.0. The plain mean gives the single window in the second share the weight of three.

## The code

```python
# gpt/workers.py
def work(pipe, shape, seed, rate):
    """One worker. Find the slopes on its share, send them, then step with everyone's average."""
    model, optimizer = start(shape, seed, rate)
    for x, y in iter(pipe.recv, None):  # until the coordinator sends None
        model.zero_grad()
        loss_on(model, x, y).backward()
        pipe.send([p.grad for p in model.parameters()])
        for p, mean in zip(model.parameters(), pipe.recv()):
            p.grad = mean
        optimizer.step()
    pipe.send(model.state_dict())


def share(pipes, x, y):
    """One step. Split the batch, gather each worker's slopes, send back their weighted average."""
    xs, ys = x.chunk(len(pipes)), y.chunk(len(pipes))
    for pipe, part, target in zip(pipes, xs, ys):
        pipe.send((part, target))
    slopes = [pipe.recv() for pipe in pipes]
    # A loss is an average over windows, so a worker with more windows counts for more.
    mean = [
        sum(len(part) * slope for part, slope in zip(xs, same)) / len(x) for same in zip(*slopes)
    ]
    for pipe in pipes:
        pipe.send(mean)
```

A pipe is a two-way channel between two processes: `send` puts an object in at one end and `recv` waits for it at the other. `iter(pipe.recv, None)` keeps receiving until a `None` arrives. `x.chunk(4)` cuts a batch into four parts along its first direction. `zip(*slopes)` lines up the same parameter's slopes from each worker.

`work` runs in a process of its own. `share` runs in the coordinator, once for each step.

## The proof

```bash
uv run pytest tests/test_workers.py
```

- Four workers on a batch of 32 windows end three steps with the weights of one process on the whole batch, to 0.00001. The four end equal to each other, to the last bit.
- A batch of 30 windows gives shares of 8, 8, 8 and 6. Counted by share, the workers still match one process.
- With the plain mean in its place, they do not.

```bash
uv run python -m gpt.workers
```

```
1 workers: 329 ms a step on a batch of 64 windows
2 workers: 254 ms a step on a batch of 64 windows
4 workers: 321 ms a step on a batch of 64 windows
```

Timings are from an M4 laptop and move from run to run.

Two workers are 1.3 times as fast as one, and four are no faster than one. The laptop has no spare cores to give them, and each step ships the slopes twice. The method is right, and this machine has nothing to gain from it. Lesson 41 runs it on four graphics chips.

## What it still cannot do

The text is clean and on disk. The size is chosen. The loop can stop and start, and the checks are written. No model has been trained on any of it yet.

## Check yourself

Predict before you run: the story model has 5,374,464 parameters. How many bytes of slopes does one worker send on one step?

<details><summary>Answer</summary>

4 bytes each: 21,497,856 bytes, about 21 MB. The same amount comes back as the average.
</details>

<details><summary>Three workers hold 10, 10 and 12 windows. What share of the average does the third worker's slope get?</summary>

12 / 32 = 0.375, against 0.3125 for each of the others.
</details>

<details><summary>Why do the workers never need to send their weights?</summary>

They start with equal weights and each takes the same step with the same average, so the weights stay equal on their own.
</details>

<details><summary>From lesson 12: the loop there also built one big batch from small ones. What is the difference here?</summary>

There the small batches ran one after another in one process. Here they run at the same time in several. The sum of slopes is the same.
</details>

Go deeper:

- PyTorch, [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html). The same scheme as PyTorch ships it, with one process for each graphics chip.
- Goyal and others, [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) (2017). What changes when the batch grows with the workers.
- Austin and others, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). How the time for sending and the time for arithmetic add up on real hardware.
