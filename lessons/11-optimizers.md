# 11. Optimizers

By the end of this page the same GPT, on the same data, in the same 3,000 steps, beats each model so far. Only the rule for the step changes.

Lesson 10's GPT scored 3.017 bits per byte. It had scored three times as many targets as an MLP a twelfth of its size, and it still trailed it.

## The idea

Gradient descent uses one rate for every parameter, and the step is rate × slope. In our untrained GPT, the typical slope in the steepest grid is about 400 times that in the flattest. So with one rate the flat grid moves 400 times less than the steep one.

Each slope also comes from one small batch, so it carries noise, as lesson 3 said. Steps point this way, then that, and partly undo each other.

An optimizer is the rule that turns slopes into steps. This page has two.

Momentum works on the noise. Keep a velocity for each parameter. On each step, multiply the velocity by 0.9 and add the new slope, then move along the velocity. Slopes that keep the same sign build up, to as much as ten times one slope. Slopes that flip sign cancel.

Adam works on the single rate. For each parameter it keeps two running averages: of the slope, and of the slope squared. The step is

rate × average slope / square root of the average squared slope

The bottom of that fraction is the slope's usual size. Dividing by it removes the scale. A parameter whose slope is steady moves by about `rate` on each step, whether that slope is 0.001 or 1,000. A parameter whose slope keeps flipping has a small average and a large usual size, so it moves little.

Both averages start at 0, which makes them too small for the first steps. Adam divides each by a factor that corrects this and fades to 1.

AdamW adds weight decay. On each step each weight grid is multiplied by 1 - rate × decay. With our rate of 0.001 and decay of 0.01 that is 0.99999. Weights that the slopes do not keep pushing up drift toward 0. Biases, gains and shifts are left alone.

## By hand

Momentum, with a rate of 0.1 and a parameter at 5. The slopes arrive as 1, 1, then -1.

| Slope | Velocity | Step | Parameter |
|---|---|---|---|
| 1 | 0.9 × 0 + 1 = 1 | 0.1 | 4.9 |
| 1 | 0.9 × 1 + 1 = 1.9 | 0.19 | 4.71 |
| -1 | 0.9 × 1.9 - 1 = 0.71 | 0.071 | 4.639 |

The third slope pointed back up, and the parameter still moved down.

Adam's first step, with a slope of 0.001. The average slope moves a tenth of the way from 0, to 0.0001. The correction factor is 1 - 0.9 = 0.1, which restores 0.001. The average square moves a thousandth of the way, to 0.000000001, and its correction restores 0.000001, whose square root is 0.001. The step is rate × 0.001 / 0.001, which is the rate. Start from a slope of 1,000 and the same sums give the rate again.

## The code

```python
# gpt/optim.py
    @torch.no_grad()
    def step(self):
        for p, velocity in zip(self.params, self.velocity):
            velocity.mul_(self.momentum).add_(p.grad)
            p -= self.rate * velocity
```

A method that ends in an underscore, such as `mul_`, changes the tensor in place. With `momentum=0` the velocity is the slope, and this is lesson 10's step.

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

`b1` is 0.9 and `b2` is 0.999. `p.dim() > 1` picks out the grids. `fix_mean` and `fix_square` are the two correction factors. `eps` is a tiny number that keeps the division safe when a slope has been exactly 0.

## The proof

```bash
uv run pytest tests/test_optim.py
```

- Our gradient descent, our momentum and our AdamW each equal PyTorch's, parameter for parameter, after each of 20 steps on a small GPT.
- The momentum example gives 4.9, 4.71 and 4.639.
- Adam's first step is the rate, for slopes of 0.001, 1 and 1,000.
- Decay shrinks a weight grid and leaves a bias alone.
- A slope that stays at 1 builds the velocity to 10, and Adam moves a flipping slope far less than a steady one.
- In the untrained GPT the typical slope differs between grids by a factor between 300 and 500.

```bash
uv run python -m gpt.optim
```

```
gradient descent: 3.017 bits per byte
with momentum: 2.564 bits per byte
AdamW: 2.372 bits per byte
```

Each run is lesson 10's model, data, seed and 3,000 steps. Only the optimizer changes. Gradient descent prints 3.017 and the same sample as lesson 10, because the same seed on `cpu` repeats.

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

## What it still cannot do

```
Once upon a time, and at words forth had,
To the tumb offench, followed abd;
Hich fless better a
```

We ran 3,000 steps at one fixed rate and stopped. Nothing looked at unseen text along the way. A run cut off at step 2,999 would have to start again from step 0.

## Check yourself

Predict before you run: with momentum of 0.9, a slope that stays at 1 forever builds the velocity toward what value?

<details><summary>Answer</summary>

10. The velocity settles where 0.9 × v + 1 = v.
</details>

<details><summary>A parameter's slope is 0.5, then -0.5, then 0.5, and so on. Roughly how far does Adam move it per step, compared with a parameter whose slope stays at 0.5?</summary>

Far less. Its average slope stays near 0 while its usual size is 0.5, so the fraction is small. The steady parameter's fraction is 1.
</details>

<details><summary>Why does weight decay skip the biases and the gains?</summary>

A gain is meant to sit near 1, and pulling it toward 0 would shrink each vector it scales. The decay is there to keep the large grids of weights small.
</details>

<details><summary>From lesson 3: a batch's slopes carry noise. Which of this page's rules averages slopes over steps, and over about how many?</summary>

Momentum, and Adam's average slope. With 0.9, a slope's share falls by a tenth on each step, so about the last ten steps carry most of the weight.
</details>

Go deeper:

- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (2014).
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (2017). The W in AdamW.
- Gabriel Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/). Momentum with sliders you can drag.
