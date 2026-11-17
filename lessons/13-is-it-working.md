# 13. Is it working?

By the end of this page you have six checks that turn a silent bug into a failed line, and a measured cure for a model that learns its text by heart.

A model with no mask still prints a falling loss and raises no error. A test here shows it: on random text, where nothing can be learned, its loss ends more than 1 nat below a healthy model's.

## The idea

A check is a fact that holds for a healthy run and fails for a broken one. We run these before a long run.

- The first loss is near ln of the vocabulary size. A first loss far from that means the starting weights are wrong.
- The stream keeps its size. Measure the spread of the residual stream after the embedding and after each block. A block adds a correction, so by the last block the spread should not have doubled.
- Each target is its input moved one token on. If the targets equal the inputs, the model learns to copy.
- No place reads the future. Change a window's last token and the scores at each earlier place must not change by a bit.
- The loop can learn one batch by heart. If the loss stays up, the loop is broken: slopes are not reaching the weights, or old slopes are piling up.
- Each grid of weights moves a little on each step. Measure how far a grid moved as a share of its own size. A share of 0 means the optimizer never got that grid. Our check fails a share above 2%.

The step check explains lesson 12's warmup. At the peak rate, Adam's first step would move a grid by 15% of its size or more, and the table below works it out. Later steps are smaller: lesson 11 showed that a slope which varies from batch to batch has an average smaller than its usual size.

A run can pass each check and still fail: the loss on the training text keeps falling while the loss on unseen text stays high. The model is storing its text. The difference between the two losses is called the gap, and we say a model with a large gap overfits. It shows when the text is small, so we train on the first 100,000 characters, a tenth of the usual.

Lesson 11's weight decay is one cure. Dropout is the other. While training, dropout sets each number a layer adds to the residual stream to 0 with a fixed chance and divides the rest by the chance of being kept, which keeps their average the same over many passes. The idea, from the paper linked below, is that a model which cannot count on any one number has to spread what it learns. After training we switch dropout off and use every number.

## By hand

A step size. Most grids start with entries about 0.02 in size, and Adam's first step moves each one by the rate.

| Rate | Share of the grid's size |
|---|---|
| 0.00003 | 0.00003 / 0.02 = 0.15% |
| 0.003 | 0.003 / 0.02 = 15% |

The two layers in each block that add into the stream start at 0.0071, so the same step is 42% of their size.

Dropout with a chance of 0.5 on the vector (1, 2, 3, 4). Suppose the first and third entries are dropped. The rest are divided by 0.5:

(0, 4, 0, 8)

The mean was 2.5 and is now 3. Averaged over each way of choosing what to drop, it is 2.5 again, because each entry is kept half the time at twice its size.

## The code

```python
# gpt/checks.py
def step_sizes(model, optimizer, x, y):
    """Take one step. For each weight grid, return how far it moved as a share of its own size."""
    grids = {name: p for name, p in model.named_parameters() if p.dim() > 1}
    before = {name: p.detach().clone() for name, p in grids.items()}
    model.zero_grad()
    loss_on(model, x, y).backward()
    optimizer.step()
    return {
        name: ((p - before[name]).norm() / before[name].norm()).item() for name, p in grids.items()
    }
```

`named_parameters` gives each parameter with its name. `.norm()` is a tensor's length: the square root of the sum of its squares. `detach().clone()` copies a tensor's numbers without its autograd history.

```python
# gpt/checks.py
def add_dropout(model, chance):
    """While training, zero a random share of what each layer adds to the residual stream."""
    model.drop = nn.Dropout(chance)  # as a part of the model, it switches off with model.eval()
    for block in model.blocks:
        for layer in (block.attn, block.narrow):
            layer.register_forward_hook(lambda layer, inputs, out: model.drop(out))
```

A forward hook is a function PyTorch calls each time a layer has run. The hook's return value replaces the layer's result. `model.eval()` puts a model in evaluation mode, where `nn.Dropout` passes its input through.

## The proof

```bash
uv run pytest tests/test_checks.py
```

A healthy model passes each check. Then the test plants seven bugs, and each turns a check red:

| Planted bug | The check that fails |
|---|---|
| Token rows 20 times too large | The first loss is far from ln of the vocabulary size |
| The layers that add into the stream 20 times too large | The stream more than doubled, and the first loss still passed |
| Targets not moved on | Targets are not the inputs one token on |
| The mask removed | An earlier place changed when the last token did |
| A first step at 100 times the warmup's rate | A grid moved by more than 2% of its size |
| Only the blocks handed to the optimizer | A grid did not move |
| Slopes never cleared | One batch is not learned by heart |

With dropout on, two passes give different results. Switched off, the model gives the result it gave before. It reaches both layers that add into the stream.

```bash
uv run python -m gpt.checks
```

```
plain: training text 0.491, unseen text 3.648, gap 3.157 nats
dropout 0.3 and decay 0.1: training text 1.230, unseen text 2.286, gap 1.055 nats
```

Both runs are lesson 12's loop for 3,000 steps on 100,000 characters. The plain run has no dropout and no decay.

Plain, the model stores its text: 0.491 nats on the text it trained on and 3.648 on unseen text, worse than the counted bigram's 2.482. With dropout and decay together the gap falls from 3.157 nats to 1.055, and the loss on unseen text to 2.286. Two things changed at once, so this measures the pair together.

## What it still cannot do

A run can pass each check after a change that did not help. The output above compares two runs, one of each setting, and another seed would give other numbers. We do not yet know how far apart two runs of the same setting land.

## Check yourself

Predict before you run: a model has a vocabulary of 1,025 tokens and good starting weights. What is its first loss?

<details><summary>Answer</summary>

About ln 1,025 = 6.93 nats.
</details>

<details><summary>The loss on the training text is 0.5 nats and on unseen text 3.6. Which tool do you reach for, and what would a check from this page say?</summary>

Dropout and weight decay, or more text. Each check would pass: the run is healthy and the model is storing its text.
</details>

<details><summary>From lesson 1: the plain run scores 3.648 nats on unseen text. What does a model that knows nothing score there, and what does that say about the plain run?</summary>

ln 65 = 4.174 nats. After 3,000 steps the plain run is half a nat better than knowing nothing, on text it has not seen.
</details>

Go deeper:

- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). The source of most of these checks.
- Srivastava and others, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (2014).
- Andrej Karpathy, [Building makemore part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc). Step sizes as a share of weight sizes, plotted for each layer.
