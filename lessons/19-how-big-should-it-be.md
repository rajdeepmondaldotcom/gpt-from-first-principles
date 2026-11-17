# 19. How big should it be?

By the end of this page a law fitted to short runs has predicted runs it never saw, and the story model has a size.

We have 52 million tokens, a model whose size we can set, and about three hours on this Mac's graphics chip. A small model can take many steps in that time. A large one can take few. We want the size with the lowest loss on unseen stories.

## The idea

Count the budget in multiplies and adds, as lesson 17 did, and not in hours. Hours depend on what else the machine is doing. A training step costs about three forward passes. Three hours at the speed lesson 17 measured is 2.7 × 10¹⁵ multiplies and adds.

Trying each size at the full budget would cost three hours a try. So we try many sizes at budgets a hundred times smaller, find the pattern in the results, and follow the pattern up.

The pattern has a known form:

loss = floor + a / parameters^alpha + b / tokens^beta

A model with few parameters is held back by its size. That is the middle term, and it shrinks as the model grows. A model that has seen few tokens is held back by its training. That is the last term. The floor is what neither removes. At a fixed budget, parameters times tokens is fixed, because each token costs about 6 multiplies and adds for each parameter. So a larger model shrinks the middle term and grows the last one. The best size is where the two pulls balance.

The law has five unknown numbers. Pick the two powers, and what is left is a weighted sum of three columns: a column of ones, one of parameters^-alpha and one of tokens^-beta. The weights that make the sum of squared misses smallest have an exact answer, called least squares, and PyTorch works it out. So we try each pair of powers from 0.10 to 1.50 and keep the pair whose best weights miss least.

Each size gets its own learning rate, found by lesson 14's sweep. The three smallest models did best at 0.01 and the three largest at 0.003. One rate for all would have held some sizes back and bent the law.

A law that fits its own runs has shown nothing. So we fit it on budgets of 1, 2 and 4 × 10¹³, 18 runs, and ask it to predict four runs at 8 × 10¹³ that it has not seen.

## By hand

The law fitted to all 22 runs has a floor of 0.119, a of 1.391, b of 0.806, alpha of 0.25 and beta of 0.55, with parameters counted in millions and tokens in tens of millions.

Take 5,374,464 parameters. The full budget buys that model 8,921 steps of 8,192 tokens, which is 73,080,832 tokens.

| Term | Sum | Value |
|---|---|---|
| Floor | | 0.119 |
| Size | 1.391 × 5.374^-0.25 = 1.391 × 0.657 | 0.913 |
| Training | 0.806 × 7.308^-0.55 = 0.806 × 0.335 | 0.270 |
| Predicted loss | | 1.302 |

Double the parameters and the size term falls by 2^-0.25, to 0.84 of itself. The tokens halve, and the training term grows by 2^0.55, to 1.46 of itself.

## The code

```python
# gpt/scale.py
def law(runs):
    """Fit loss = floor + a / parameters^alpha + b / tokens^beta to (parameters, tokens, loss) runs.

    For each pair of powers, least squares gives the best floor, a and b. Keep the best pair.
    """
    n, d, loss = (torch.tensor(column, dtype=torch.float64) for column in zip(*runs))
    best = None
    for alpha in POWERS:
        for beta in POWERS:
            # Parameters in millions and tokens in tens of millions keep each column near 1.
            grid = torch.stack([torch.ones_like(n), (n / 1e6) ** -alpha, (d / 1e7) ** -beta], dim=1)
            parts = torch.linalg.lstsq(grid, loss).solution
            miss = (grid @ parts - loss).pow(2).mean().sqrt().item()
            if parts.min() >= 0 and (best is None or miss < best["miss"]):
                best = {"alpha": alpha, "beta": beta, "parts": parts.tolist(), "miss": miss}
    return best
```

`zip(*runs)` turns a list of (parameters, tokens, loss) into three columns. `torch.float64` holds each number in 64 bits, which this sum needs. `miss` is the root of the mean squared miss: the usual distance between the law and a run. A fit with a negative floor, a or b is thrown away.

## The proof

```bash
uv run pytest tests/test_scale.py
```

- The hand rule gives floor + a + b at 1 million parameters and 10 million tokens.
- Runs made from a law we choose give back that law: both powers, the floor, a and b.
- With noise of 0.02 nats added, the fit moves a little and its miss reports the noise.
- At a fixed budget one size is best, and 16 times the budget moves it by 16 to the power beta / (alpha + beta).
- A run spends its budget, and a bigger model gets fewer tokens from it.

The script takes over an hour on the graphics chip. Part of what it prints is below.

```bash
uv run python -m gpt.scale
```

```
width 32: rate 0.001 gives 3.352, 0.003 gives 3.004, 0.01 gives 2.852
width 48: rate 0.001 gives 3.313, 0.003 gives 2.874, 0.01 gives 2.753
width 64: rate 0.001 gives 3.292, 0.003 gives 2.819, 0.01 gives 2.736
width 96: rate 0.001 gives 3.388, 0.003 gives 2.998, 0.01 gives 3.577
width 128: rate 0.001 gives 3.556, 0.003 gives 3.233, 0.01 gives 3.946
width 192: rate 0.001 gives 3.728, 0.003 gives 3.708, 0.01 gives 4.680
4e+13 flops,   155,776 parameters, 35,389,440 tokens: 2.774 nats
4e+13 flops,   252,192 parameters, 22,142,976 tokens: 2.603 nats
4e+13 flops,   352,384 parameters, 15,966,208 tokens: 2.493 nats
4e+13 flops,   688,896 parameters,  7,979,008 tokens: 2.538 nats
4e+13 flops, 1,245,952 parameters,  4,423,680 tokens: 2.677 nats
4e+13 flops, 2,410,368 parameters,  2,375,680 tokens: 3.036 nats
law from 18 runs: floor 0.12, alpha 0.30, beta 0.50, misses its own runs by 0.195 nats
8e+13 flops,   352,384 parameters, 31,932,416 tokens: 2.383 nats
8e+13 flops,   688,896 parameters, 15,958,016 tokens: 2.271 nats
8e+13 flops, 1,245,952 parameters,  8,847,360 tokens: 2.301 nats
8e+13 flops, 2,410,368 parameters,  4,759,552 tokens: 2.463 nats
held out,   352,384 parameters: predicted 2.285, off by +0.098
held out,   688,896 parameters: predicted 2.214, off by +0.057
held out, 1,245,952 parameters: predicted 2.261, off by +0.040
held out, 2,410,368 parameters: predicted 2.436, off by +0.027
law from all 22 runs: floor 0.12, alpha 0.25, beta 0.55, misses its own runs by 0.177 nats
 1,245,952 parameters, 36,452 steps: predicted 1.560 nats
 2,410,368 parameters, 19,605 steps: predicted 1.410 nats
 5,374,464 parameters,  8,921 steps: predicted 1.302 nats
10,325,120 parameters,  4,723 steps: predicted 1.278 nats
14,555,904 parameters,  3,407 steps: predicted 1.289 nats
```

The law passed its test. Fitted on 18 runs, it predicted four runs at twice the largest budget to within 0.027 to 0.098 nats, each a little too low. It also named the best of the four sizes: 688,896 parameters, predicted lowest and measured lowest.

It misses its own runs by more, 0.195 nats. The shortest runs are rough: at the smallest budget the 352,384-parameter model scored 3.974 nats, worse than the models on both sides of it.

Now the size. At the full budget the predictions fall to 1.278 nats at 10 million parameters and rise after. The bottom is flat: from 5 to 15 million parameters they differ by 0.024 nats, and the law missed its held-out runs by up to 0.098. It cannot tell these three sizes apart. We take the smallest of them, 5,374,464 parameters, because a smaller model costs less to fine-tune and to serve.

One caution. The full budget is 34 times the largest the law has seen, and the test above reached 2 times.

## What it still cannot do

The run we picked is 8,921 steps on one graphics chip. Text at the scale of the web does not fit one machine's day, and the loop so far uses one process.

## Check yourself

Predict before you run: with an alpha of 0.25, how many times more parameters halve the size term?

<details><summary>Answer</summary>

Sixteen. 16^-0.25 = 0.5. A small power means size buys loss slowly.
</details>

<details><summary>Why is the budget counted in multiplies and adds and not in minutes?</summary>

A count is the same on any day and any machine. Minutes change with whatever else is running, and a law fitted to minutes would not hold tomorrow.
</details>

<details><summary>The law predicts 1.278 nats for one size and 1.302 for another. Which should we train?</summary>

The law cannot say. It missed runs it had not seen by up to 0.098 nats, four times that difference. Something else has to decide, and here it is cost.
</details>

<details><summary>From lesson 14: each size used the best of three rates. What would one rate for all sizes have done to the comparison?</summary>

It would have changed two things at once, the size and how well the rate suits it. A size could then look bad because of its rate.
</details>

Go deeper:

- Kaplan and others, [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020). The law in this form.
- Hoffmann and others, [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (2022). The same question at a budget ten billion times ours.
- Stanford CS336, [Assignment 3: scaling](https://github.com/stanford-cs336/assignment3-scaling). Fit a law yourself, with a budget of runs to spend.
