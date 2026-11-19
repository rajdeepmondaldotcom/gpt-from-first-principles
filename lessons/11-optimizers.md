# 11. Optimizers

By the end of this page the same GPT, on the same data, in the same 3,000 steps, beats each model so far, with a new rule for the step.

Lesson 10's GPT scored 3.017 bits per byte. It had scored three times as many targets as an MLP a twelfth of its size, and it still trailed it.

## The idea

Gradient descent uses one rate for every parameter, and the step is rate × slope. Take the average size of the slopes in each tensor of our untrained GPT's parameters. Among the weight grids, the steepest averages 10 to 20 times the flattest. Across all tensors, a bias against a gain, the gap is 300 to 500 times. With one rate, the flattest moves hundreds of times less than the steepest.

Each slope also comes from one small batch, so it carries noise. Steps point this way, then that, and undo some of each other's work.

An optimizer is the rule that turns slopes into steps.

Momentum works on the noise. Keep a velocity for each parameter. On each step, multiply the velocity by 0.9 and add the new slope, then move along the velocity. Slopes that keep the same sign build up, to as much as ten times one slope. Slopes that flip sign cancel.

Adam works on the single rate. For each parameter it keeps two running averages: of the slope, and of the slope squared. A running average moves a fixed share of the way toward each new value. The average slope moves a tenth of the way: 0.9 × old average + 0.1 × new slope. The average square moves a thousandth of the way. The step is

rate × average slope / square root of the average squared slope

The bottom of that fraction is the slope's usual size. Dividing by it removes the scale. A parameter whose slope is steady moves by about `rate` on each step, whether that slope is 0.001 or 1,000. A parameter whose slope keeps flipping has a small average and a large usual size, so it moves little.

Both averages start at 0, which makes them too small for the first steps. So after t steps Adam divides the average slope by 1 - 0.9 to the power t, and the average square by 1 - 0.999 to the power t.

AdamW adds weight decay. On each step AdamW multiplies each weight grid by 1 - rate × decay. A weight that the slopes do not keep pushing away from 0 drifts back toward it. Adding decay to the loss would send its slope through Adam's division, and weights with large slopes would hardly decay. AdamW shrinks the weights outside the division. We leave biases, gains and shifts alone. Lesson 13 measures decay, paired with another change.

## By hand

Momentum, with a rate of 0.1 and a parameter at 5. The slopes arrive as 1, 1, then -1.

| Slope | Velocity | Step | Parameter |
|---|---|---|---|
| 1 | 0.9 × 0 + 1 = 1 | 0.1 | 4.9 |
| 1 | 0.9 × 1 + 1 = 1.9 | 0.19 | 4.71 |
| -1 | 0.9 × 1.9 - 1 = 0.71 | 0.071 | 4.639 |

Adam's first step, with a slope of 0.001. The average slope moves a tenth of the way from 0, to 0.0001. The correction factor is 1 - 0.9 = 0.1, which restores 0.001. The average square moves a thousandth of the way, to 0.000000001, and its correction restores 0.000001, whose square root is 0.001. The step is rate × 0.001 / 0.001, which is the rate.

| Adam's first step | Slope 0.001 | Slope 1,000 |
|---|---|---|
| Average slope, corrected | 0.001 | 1,000 |
| √ average square, corrected | 0.001 | 1,000 |
| Step | rate | rate |

## The code

```python
# gpt/optim.py
    @torch.no_grad()
    def step(self):
        for p, velocity in zip(self.params, self.velocity):
            velocity.mul_(self.momentum).add_(p.grad)
            p -= self.rate * velocity
```

`@torch.no_grad()` above a function runs the whole function without recording. A method that ends in an underscore, such as `mul_`, changes the tensor in place. With `momentum=0` the velocity is the slope, and this is lesson 10's step.

```python
# gpt/optim.py
        for p, mean, square in zip(self.params, self.mean, self.square):
            if p.dim() > 1:
                p *= 1 - self.rate * self.decay  # pull weight grids toward 0; leave the rest
            mean.lerp_(p.grad, 1 - b1)  # move the average a tenth of the way to the new slope
            square.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            size = square.sqrt() / math.sqrt(fix_square) + self.eps  # the slope's usual size
            p.addcdiv_(mean, size, value=-self.rate / fix_mean)  # p -= rate * mean / size
```

`b1` is 0.9 and `b2` is 0.999. `p.dim() > 1` picks out the grids. `fix_mean` is 1 - 0.9 to the power t and `fix_square` is 1 - 0.999 to the power t. `addcmul_(g, g, value=c)` adds c × g × g in place. `eps` keeps the division safe when a slope has been 0. Adam keeps two numbers per parameter, `mean` and `square`, so with the parameters and their slopes, training holds four copies of the model's size.

## The proof

```bash
uv run pytest tests/test_optim.py
```

- Our gradient descent, momentum and AdamW each equal PyTorch's, parameter for parameter, after each of 20 steps on a small GPT. AdamW also matches with other settings.
- The momentum example gives 4.9, 4.71 and 4.639.
- Adam's first step is the rate, for slopes of 0.001, 1 and 1,000.
- Decay shrinks a weight grid and leaves a bias alone.
- A slope that stays at 1 builds the velocity to 10, and Adam moves a flipping slope far less than a steady one.
- The slope sizes above hold in the untrained GPT.

Predict before you run: does momentum close more or less than half the gap between gradient descent and AdamW?

```bash
uv run python -m gpt.optim
```

```
gradient descent: 3.017 bits per byte
with momentum: 2.564 bits per byte
AdamW: 2.372 bits per byte
lesson 4's MLP reading 8 characters back, with AdamW: 2.551 bits per byte
lesson 4's MLP reading 64 characters back, with AdamW: 2.863 bits per byte
```

AdamW takes the GPT from 3.017 to 2.372 bits per byte on unseen text.

The first three runs are lesson 10's model, data, seed and 3,000 steps, with only the optimizer changed. AdamW's rate is 0.001, its decay 0.01. Momentum keeps gradient descent's rate of 0.1, so its steady steps are up to ten times larger, and part of its gain here may be the larger step. A seed starts the random number generator: it fixes the starting weights and the batches, and on `cpu` it repeats.

The last two lines give lesson 4's MLP the same AdamW, steps and count of targets. Reading 64 places as the GPT does, the MLP does worse than reading 8, with 64 sets of weights to train in place of 8.

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
| MLP, eight characters back, with AdamW | 2.551 |
| MLP, 64 characters back, with AdamW | 2.863 |

## What it still cannot do

```
Once upon a time, and at words forth had,
To the tumb offench, followed abd;
Hich fless better a
```

We ran 3,000 steps at one fixed rate and stopped. The run never measured unseen text along the way. A run cut off at step 2,999 would have to start again from step 0.

## Check yourself

<details><summary>Prediction</summary>

More: 2.564 closes 0.453 of the gap of 0.645, 70%.
</details>

<details><summary>Momentum is 0.9 and the slope stays at 1. Where does the velocity settle?</summary>

10, where 0.9 × v + 1 = v.
</details>

<details><summary>A parameter's slope is 0.5, then -0.5, then 0.5, and so on. Roughly how far does Adam move it per step, compared with a parameter whose slope stays at 0.5?</summary>

About a twentieth as far. Its average slope swings between about 0.026 and -0.026 while its usual size is 0.5.
</details>

<details><summary>From lesson 3: a batch's slopes carry noise. Which of this page's rules averages slopes over steps, and over about how many?</summary>

Momentum, and Adam's average slope. With 0.9, about the last ten steps carry most of the weight.
</details>

Go deeper:

- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (2014).
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (2017). The W in AdamW.
- Gabriel Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/). Momentum with sliders you can drag.
