# 13. Is it working?

By the end of this page you have five checks that turn a silent bug into a failed line, and a measured cure for a model that learns its text by heart.

Lesson 12's loop prints numbers. A model with no mask still trains, prints a falling loss, and raises no error. A test here shows it: on random text its loss ends more than 1 nat below a healthy model's. Nothing yet says whether the numbers are the ones a healthy run would print.

## The idea

A check is a fact that holds for a healthy run and fails for a broken one. Each of these takes seconds, so we run them before a long run.

- The first loss is near ln of the vocabulary size. Lesson 10's starting weights give each token about the same chance. A first loss far from that means the starting weights are wrong.
- Each target is its input moved one token on. If the targets equal the inputs, the model learns to copy, and its loss falls to nearly 0 for the wrong reason.
- No place reads the future. Change the last token of a window and the scores at each earlier place must stay the same to the last bit.
- The loop can learn one batch by heart. A model with far more parameters than the batch has tokens can store the answers. If the loss stays up, slopes are not reaching the weights.
- Each grid of weights moves a little on each step. Measure how far a grid moved as a share of its own size. A share of 0 means the optimizer never got that grid. A share above 2% means the rate is too high for it.

The last check explains lesson 12's warmup. Adam's first step moves each entry by the rate. Most of our grids start with entries about 0.02 in size, so a first step at the peak rate of 0.003 moves a grid by about 15% of its size. At the warmup's first rate, 0.00003, it moves by about 0.15%.

A healthy run can still fail in one way that no check above sees: the loss on the training text falls while the loss on unseen text rises. The model is storing its text. The difference between the two losses is called the gap, and a growing gap is called overfitting. It shows when the text is small. We train on the first 100,000 characters, a tenth of the usual text, to see it.

Two tools shrink the gap.

Weight decay is lesson 11's: each step pulls the weight grids toward 0.

Dropout is new. While training, set a random share of the numbers each layer adds to the residual stream to 0, and divide the rest by the share that is left, which keeps their average the same. A different random share is dropped on each pass. After training, dropout is switched off and the whole model answers.

## By hand

A step size. A grid's entries are about 0.02 in size, and Adam's first step moves each one by the rate.

| Rate | Share of the grid's size |
|---|---|
| 0.00003 | 0.00003 / 0.02 = 0.15% |
| 0.003 | 0.003 / 0.02 = 15% |

Dropout with a share of 0.5 on the vector (1, 2, 3, 4). Suppose the first and third entries are dropped. The rest are divided by 0.5:

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

A forward hook is a function PyTorch calls each time a layer has run. What the hook returns takes the place of the layer's result. So dropout goes onto lesson 10's model from outside, and `gpt/gpt2.py` does not change. `model.eval()` puts a model in evaluation mode, where `nn.Dropout` passes its input through. `model.train()` switches it back.

## The proof

```bash
uv run pytest tests/test_checks.py
```

A healthy small model passes each check. Then six bugs are planted, one at a time, and each turns a check red:

| Planted bug | The check that fails |
|---|---|
| Token rows 20 times too large | The first loss is far from ln of the vocabulary size |
| Targets not moved on | Targets are not the inputs one token on |
| The mask removed | An earlier place changed when the last token did |
| A first step at 100 times the warmup's rate | A grid moved by more than 2% of its size |
| The table of places left out of the optimizer | A grid did not move |
| Slopes never cleared | One batch is not learned by heart |

Dropout changes the result while training, differently on each pass, and changes nothing once it is switched off.

```bash
uv run python -m gpt.checks
```

```
plain: training text 0.491, unseen text 3.648, gap 3.157 nats
dropout 0.3 and decay 0.1: training text 1.230, unseen text 2.286, gap 1.055 nats
```

Both runs are lesson 12's loop for 3,000 steps on 100,000 characters.

Plain, the model stores its text: 0.491 nats on the text it trained on and 3.648 on unseen text, worse than the bigram. With dropout and decay the gap falls from 3.157 nats to 1.055, and the loss on unseen text from 3.648 to 2.286.

## What it still cannot do

The checks say a run is healthy. They cannot say whether a change made it better. The output above compares two runs, one of each setting. Another seed would give other numbers. We do not yet know how far apart two runs of the same setting land.

## Check yourself

Predict before you run: a model has a vocabulary of 1,025 tokens and good starting weights. What is its first loss?

<details><summary>Answer</summary>

About ln 1,025 = 6.93 nats.
</details>

<details><summary>The loss on the training text is 0.5 nats and on unseen text 3.6. Which tool do you reach for, and what would a check from this page say?</summary>

Dropout and weight decay, or more text. The checks would all pass. The run is healthy and the model is storing its text.
</details>

<details><summary>Dropout with a share of 0.2 keeps a number. What does it become?</summary>

It is divided by 0.8, so it grows by a quarter.
</details>

<details><summary>From lesson 11: why does a first step at the peak rate move each grid by about the same share of its size, whatever its slopes?</summary>

Adam's first step is the rate for each parameter, whatever the slope. Each grid starts with entries of about the same size, 0.02.
</details>

Go deeper:

- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). The source of most of these checks.
- Srivastava and others, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (2014).
- Andrej Karpathy, [Building makemore part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc). Step sizes as a share of weight sizes, plotted for each layer.
