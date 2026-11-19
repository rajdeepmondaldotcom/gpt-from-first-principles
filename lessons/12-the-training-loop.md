# 12. The training loop

By the end of this page you have a run you can leave. Its rate changes as it goes and it reports on unseen text. After a stop it carries on bit for bit.

Lesson 11 ran 3,000 steps at one fixed rate. Nothing looked at unseen text along the way, and a run cut off at step 2,999 would have started again from step 0.

## The idea

The rate changes as the run goes. Adam's first step moves each parameter by the full rate, whatever its slope, on the word of a single batch. So we begin with a small rate and raise it in a straight line over the first 100 steps. This is called warmup. After it, the rate falls to a tenth of its peak along half a cosine wave: a gentle fall at first, steepest in the middle, gentle at the end. The plan for the rate is called a schedule. We take warmup and a falling rate on trust here. Lesson 14 tests them against a fixed rate.

A batch can produce a gradient far longer than usual, so we put a limit on it. Adam's average of squares moves a thousandth of the way each step (lesson 11), so it would keep a trace of a long one for about a thousand steps, and divide each step in that time by a larger size. Before each step we measure the length of the whole gradient: square each slope, add them all, take the square root. If the length is above the limit, we scale the gradient down to the limit and keep its direction. This is called clipping.

Slopes add, as lesson 2 showed, so small batches can stand in for a big one. Run four small batches, let their slopes add up, then take one step. The step is the same as for one batch four times the size, which might not fit in memory.

Every 500 steps the loop measures the loss on unseen text.

A checkpoint lets the run stop and start. It is a file that holds what shapes the rest of this run:

- the model's parameters
- the optimizer's two averages for each parameter, and its count of steps
- the step number, because the schedule depends on it
- the state of the random number generator. A generator makes its numbers by a fixed rule from a state it keeps, so the state sets the batches to come

The loop has a switch for 16-bit numbers. PyTorch can run the pass from tokens to loss, called the forward pass, with each number held in 16 bits in place of 32, while the weights and the optimizer stay in 32. On a graphics chip, or GPU, this saves memory and time. Here the switch stays off. Its format, `bfloat16`, keeps the range of a 32-bit number and gives up digits, so tiny slopes do not round to 0 as they can in other 16-bit formats.

The run below uses a limit of 1 and batches of 32 windows.

## By hand

A schedule with a peak of 1, a warmup of 10 steps and 100 steps in all.

| Step | Rate | Why |
|---|---|---|
| 0 | 0.1 | one tenth of the way up |
| 9 | 1.0 | the peak |
| 55 | 0.55 | halfway through the fall, halfway between 1.0 and 0.1 |
| 99 | 0.1003 | a little above the floor of 0.1 |

Clipping: two parameters have slopes 3 and 4. The gradient's length is the square root of 9 + 16, which is 5. With a limit of 1, each slope is multiplied by 1/5, giving 0.6 and 0.8.

## The code

```python
# gpt/train.py
def rate_at(step, peak, warmup, steps, floor=0.1):
    """The learning rate for one step: a straight climb to `peak`, then a slow curve down."""
    if step < warmup:
        return peak * (step + 1) / warmup
    done = (step - warmup) / max(1, steps - warmup)  # 0 at the peak, nearly 1 at the last step
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * done)))
```

```python
# gpt/train.py
    for step in range(start, steps):
        optimizer.rate = rate_at(step, peak, warmup, steps, floor)
        model.zero_grad()
        total = 0.0
        for _ in range(parts):  # a big batch, built from `parts` small ones that fit in memory
            with torch.autocast(device, dtype=torch.bfloat16, enabled=half):
                loss = loss_on(model, *batch(ids, model.context, size, generator)) / parts
            loss.backward()
            total += loss.item()
        clip(model.parameters(), limit)
        optimizer.step()
        yield step + 1, total
```

`yield` hands the step number and the loss back to whoever called `train`, then carries on when asked for the next one. We divide each small loss by `parts` because a loss is an average, and the average over the big batch is the average of the small averages. `torch.autocast` is the 16-bit switch, off unless `half` is set.

```python
# gpt/train.py
    state = {
        "model": model.state_dict(),
        "mean": optimizer.mean,
        "square": optimizer.square,
        "steps": optimizer.steps,
        "step": step,
        "generator": generator.get_state(),
    }
```

`state_dict()` is a dictionary of a model's parameters by name. `torch.save` writes the whole dictionary to a file.

## The proof

```bash
uv run pytest tests/test_train.py
```

- The rate climbs in a straight line to its peak, then falls without a bump to the floor. With a floor of 1 and a warmup of one step it is a fixed rate.
- The clipping example gives 0.6 and 0.8, and a gradient under the limit is left alone.
- The loop can learn one batch by heart: on a single window the loss falls from above 2 to under 0.05.
- Ten steps, a save, a load into a fresh model and ten more steps end with each parameter equal, to the last bit, to 20 steps unbroken.
- Four small batches through the loop give the same step and the same loss as one big batch.
- The loop clips before it steps: with a limit of 0.01 and a rate of 1, one step moves the weights by 0.01.
- Training in 16 bits ends within 0.05 nats of training in 32, and not at the same number.

Predict before you run: from step 4,500 to 5,000 the rate sits near its floor. Does the loss on unseen text still fall?

```bash
uv run python -m gpt.train
```

```
step 500: train 1.881, unseen 1.994 nats, rate 0.00296
step 1000: train 1.638, unseen 1.826 nats, rate 0.00278
step 1500: train 1.506, unseen 1.732 nats, rate 0.00249
step 2000: train 1.548, unseen 1.679 nats, rate 0.00212
step 2500: train 1.420, unseen 1.632 nats, rate 0.00169
step 3000: train 1.440, unseen 1.608 nats, rate 0.00127
step 3500: train 1.375, unseen 1.583 nats, rate 0.00088
step 4000: train 1.309, unseen 1.557 nats, rate 0.00057
step 4500: train 1.259, unseen 1.541 nats, rate 0.00037
step 5000: train 1.291, unseen 1.529 nats, rate 0.00030
unseen text: 1.529 nats, 2.206 bits per byte
```

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| Bigram, learned | 3.599 |
| MLP, eight characters back | 2.663 |
| MLP, eight tokens back, 1,024 tokens | 2.572 |
| GPT, 64 characters back | 3.017 |
| GPT, with momentum | 2.564 |
| GPT, with AdamW | 2.372 |
| GPT, with the loop, 5,000 steps | 2.206 |

The loop takes the GPT to 2.206 bits per byte on unseen text. Between the last two rows four things changed: 5,000 steps, not 3,000; a peak of 0.003, not a fixed 0.001; a decay of 0.1, not 0.01; and the schedule with clipping.

## What it still cannot do

```
Once upon a time of men, in backling all an embrace
As your o'er peace will keep again the sword,
So many foot, your worshonour that my
```

From the printed numbers alone you cannot tell a healthy run from a broken one. A model with no mask still prints a falling loss and raises no error.

## Check yourself

<details><summary>Prediction</summary>

Yes, from 1.541 to 1.529 nats. The training loss, from one batch, rose from 1.259 to 1.291.
</details>

<details><summary>The peak is 0.003 and the warmup is 100 steps. What is the rate on the first step?</summary>

0.003 × 1 / 100 = 0.00003.
</details>

<details><summary>A checkpoint leaves out the generator's state. What happens on resume?</summary>

It draws other batches than the unbroken run, so the two no longer match.
</details>

<details><summary>From lesson 2: what happens if the loop never sets the slopes back to 0?</summary>

Each new slope is added to the old ones, and a step follows their sum.
</details>

Go deeper:

- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). Learn one batch by heart first, and the habits that follow.
- Loshchilov and Hutter, [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). Where the cosine curve comes from.
- PyTorch, [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html). The 16-bit forward pass.
