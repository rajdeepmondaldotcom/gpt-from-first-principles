# 12. The training loop

By the end of this page you have a run you can leave. Its rate changes as it goes, it reports on unseen text, and after a stop it carries on from where it was, bit for bit.

Lesson 11 ran 3,000 steps at one fixed rate. Nothing looked at unseen text along the way, and a run cut off at step 2,999 would have started again from step 0.

## The idea

Five things go around the step.

A rate that changes. Adam's first step moves each parameter by the full rate, whatever its slope, and its averages hold one sample each. So we begin with a small rate and raise it in a straight line over the first 100 steps. This is called warmup. After it, the rate falls along half a cosine wave to a tenth of its peak. The plan for the rate is called a schedule. Warmup and a falling rate are common practice, and we take them on trust here. Lesson 14 tests them against a fixed rate.

A limit on the gradient. A batch can produce a gradient far longer than usual, and Adam's average of squares would hold on to it for about a thousand steps. Before each step we measure the length of the whole gradient: square each slope, add them all, take the square root. If the length is above a limit, we scale the gradient down to the limit and keep its direction. This is called clipping.

A big batch made from small ones. Slopes add, as lesson 2 showed. Run four small batches, let their slopes add up, then take one step. The step is the same as for one batch four times the size, which might not fit in memory.

A look at unseen text. The loss on the training batches says how well the model fits what it sees. Every 500 steps we also measure the loss on the unseen text.

A way to stop and start. A file called a checkpoint holds each thing that shapes the rest of the run:

- the model's parameters
- the optimizer's two averages for each parameter, and its count of steps
- the step number, because the schedule depends on it
- the state of the random number generator, which decides the batches to come

The loop also has a switch for 16-bit numbers. PyTorch can run the forward pass with each number held in 16 bits in place of 32, while the weights and the optimizer stay in 32. GPUs compute faster that way, as the PyTorch page linked below explains. On this page the switch stays off.

## By hand

A schedule with a peak of 1, a warmup of 10 steps and 100 steps in all.

| Step | Rate | Why |
|---|---|---|
| 0 | 0.1 | one tenth of the way up |
| 9 | 1.0 | the peak |
| 55 | 0.55 | halfway through the fall, halfway between 1.0 and 0.1 |
| 99 | 0.1003 | the floor |

Clipping: two parameters have slopes 3 and 4. The gradient's length is the square root of 9 + 16, which is 5. With a limit of 1, each slope is multiplied by 1/5, giving 0.6 and 0.8. The length is now 1 and the direction is unchanged.

## The code

```python
# gpt/train.py
def rate_at(step, peak, warmup, steps, floor=0.1):
    """The learning rate for one step: a straight climb to `peak`, then a slow curve down."""
    if step < warmup:
        return peak * (step + 1) / warmup
    done = (step - warmup) / max(1, steps - warmup)  # 0 at the peak, 1 at the last step
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

`yield` hands the step number and the loss back to whoever called `train`, then carries on when asked for the next one. The caller decides what to print and when to save. Each small loss is divided by `parts` because a loss is an average, and the average over the big batch is the average of the small averages. `torch.autocast` is the 16-bit switch, off unless `half` is set.

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
- A run that stops and resumes is the same run. One model trains 20 steps. Another trains 10 and saves, and a third loads the file and trains 10 more. The first and third end with each parameter equal to the last bit.
- Four small batches added up give the same slopes as one big batch.
- Training in 16 bits ends within 0.05 nats of training in 32.

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

## What it still cannot do

```
Once upon a time of men, in backling all an embrace
As your o'er peace will keep again the sword,
So many foot, your worshonour that my
```

The loop prints numbers. Nothing in it says whether they are the numbers a healthy run would print. A model with no mask still trains, prints a falling loss, and raises no error.

## Check yourself

Predict before you run: the peak is 0.003 and the warmup is 100 steps. What is the rate on the first step?

<details><summary>Answer</summary>

0.003 × 1 / 100 = 0.00003.
</details>

<details><summary>Slopes of 6 and 8, with a limit of 5. What are they after clipping?</summary>

The length is 10, so each is halved: 3 and 4.
</details>

<details><summary>A checkpoint leaves out the generator's state. What happens on resume?</summary>

The run continues and still learns, but it draws other batches than the uninterrupted run would have. The two no longer match step for step.
</details>

<details><summary>From lesson 11: why does a small first rate suit Adam?</summary>

Adam's first step moves each parameter by the full rate, whatever its slope, and its averages hold one sample each. A small rate keeps those first steps small.
</details>

Go deeper:

- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). Learn one batch by heart first, and the habits that follow.
- Loshchilov and Hutter, [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). Where the cosine curve comes from.
- PyTorch, [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html). The 16-bit forward pass.
