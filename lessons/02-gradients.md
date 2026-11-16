# 2. Gradients

By the end of this page you have a model that starts out knowing nothing and teaches itself the table that lesson 1 had to count.

Counting worked because the model was a table. For a model of any other shape there is nothing to count. We need a method that starts with wrong numbers and improves them.

## The idea

A model is a recipe with adjustable numbers in it. The numbers are called parameters. Give the bigram one parameter per pair of characters, a score, and start them all at 0.

A score can be any number, so we need a rule that turns a row of scores into chances. Raise e to the power of each score, then divide by the row's total. The results are positive and sum to 1. The rule is called softmax. With all scores at 0, each character gets the same chance.

Pick one score and ask: if I raise it a little, does the loss rise or fall, and how fast? The answer is a slope. A slope of -2 means the loss falls by about 0.02 when the score rises by 0.01. The slopes of the loss for all parameters, taken together, are the gradient.

Learning is one rule, repeated:

new value = old value - rate × slope

A positive slope means raising the parameter hurts, so we lower it. The rate, called the learning rate, sets the size of the step. Repeating this rule is called gradient descent.

The obvious way to get a slope is to nudge. Add 0.000001 to one score, recompute the loss, and divide the change by 0.000001. That costs one full loss calculation per parameter, on each step. Our sentence model has 100 scores. The GPT we build later has millions.

The chain rule gets all the slopes from one pass. A loss is built from small operations: add, multiply, exp, log. Each one knows its own slope. If x moves y at a rate of 3, and y moves the loss at a rate of 2, then x moves the loss at a rate of 6. Multiply along a path. If x reaches the loss by two paths, add them.

So we compute the loss once and let each number remember what it was made from. Then we walk backwards from the loss and hand each number its share. Code that does this is called automatic differentiation, or autograd. Ours is under 60 lines.

## By hand

Take L = a × b + c with a = 2, b = -3, c = 10. Then L = 4.

Name the middle step d = a × b = -6, so L = d + c.

- c moves L one for one. Its slope is 1. So is d's.
- a moves d at a rate of b, which is -3. In a product, the slope for one factor is the other factor. d moves L at a rate of 1. Along the path: -3 × 1 = -3.
- b moves d at a rate of a, which is 2. Its slope is 2.

Check a by nudging. With a = 2.001, L = 2.001 × -3 + 10 = 3.997. L moved by -0.003 for a nudge of 0.001. The slope is -3.

Now walk downhill by hand. Let the loss be (w - 3)², whose slope is 2 × (w - 3). Start at w = 0 with a rate of 0.1.

| Step | w | Slope | New w |
|---|---|---|---|
| 1 | 0 | -6 | 0.6 |
| 2 | 0.6 | -4.8 | 1.08 |
| 3 | 1.08 | -3.84 | 1.464 |

Each step closes a fifth of the gap to 3.

## The code

A `Value` holds a number, the Values it was made from, and how fast it moves when each of them moves:

```python
# gpt/autograd.py
class Value:
    """A number that remembers how it was made."""

    def __init__(self, data, parents=(), slopes=()):
        self.data = data
        self.grad = 0.0  # how much the final result moves when this number moves
        self.parents = parents  # the Values this one was computed from
        self.slopes = slopes  # how much this number moves when each parent moves

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
```

`backward` lists each Value after the Values it was made from, then walks that list in reverse. Each Value multiplies its own `grad` by a local slope and adds the result to its parent. That is "multiply along a path, add across paths":

```python
# gpt/autograd.py
        self.grad = 1.0
        for v in reversed(order):
            for parent, slope in zip(v.parents, v.slopes):
                parent.grad += slope * v.grad
```

The model and the learning loop:

```python
# gpt/autograd.py
def softmax(scores):
    """Turn any list of scores into chances: all positive, summing to 1."""
    exps = [s.exp() for s in scores]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]
```

```python
# gpt/autograd.py
    for _ in range(steps):
        loss = bigram_loss(W, pairs)
        loss.backward()
        for row in W:
            for w in row:
                w.data -= rate * w.grad
                w.grad = 0.0
        losses.append(loss.data)
```

## The proof

```bash
uv run pytest tests/test_autograd.py
```

- Our gradients equal those of PyTorch's autograd, on an expression that uses each operation.
- The chain rule agrees with nudging.
- The hand examples give -3, 2, 1, and the path 0.6, 1.08, 1.464.
- A number used twice collects both paths: in `a + a`, the slope for `a` is 2.
- A rate of 1.1 on the hand example runs away from 3.
- The learned table starts at ln 10 and ends within 0.01 nats of the counted one, without passing it.

```bash
uv run python -m gpt.autograd
```

```
counted table:      0.5152 nats
learned, step   0:  2.3026 nats
learned, step   1:  1.9895 nats
learned, step  10:  0.8852 nats
learned, step 100:  0.5404 nats
learned, step 299:  0.5231 nats
```

The learned model closes in on the counted one. It cannot pass it, because lesson 1 proved that no table gives the training text a higher probability. It cannot land on it in a finite number of steps either. The counted table holds chances of 0 and 1, and softmax reaches those only with scores of infinite size.

## What it still cannot do

It learns one sentence. A step on 2,000 pairs of Shakespeare takes between a third of a second and a second on an M4 laptop. The file has 1,115,393 pairs, so one step on all of it would take minutes, and learning takes hundreds of steps.

Each `Value` is a Python object holding one number. A step builds one for each addition, each multiplication and each exp.

## Check yourself

Predict before you run: after 300 steps, is the learned loss above, below or equal to the counted 0.5152?

<details><summary>What is the slope of x³ at x = 2? Check by nudging.</summary>

3 × 2² = 12. With x = 2.001, x³ = 8.012006. The change, 0.012006, divided by 0.001 is about 12.
</details>

<details><summary>In `a * a` with a = 3, why is the slope for a equal to 6 and not 3?</summary>

a reaches the result by two paths, once as each factor. Each path has slope 3, and the paths add.
</details>

<details><summary>With a rate of 1.1 on the loss (w - 3)², what happens?</summary>

Each step moves w by 2.2 times its distance from 3, so it lands on the far side, 1.2 times further away than it began. The distance grows on each step.
</details>

<details><summary>From lesson 1: the loss of a blind guess over 10 characters is ln 10. Where does that number appear on this page?</summary>

Step 0. All scores are 0, softmax gives each character 1/10, and the surprise is ln 10 = 2.3026.
</details>

Go deeper:

- Andrej Karpathy, [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0). The same engine, built live.
- 3Blue1Brown, [Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus). The chain rule in pictures.
- Baydin and others, [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767). Forward mode, reverse mode, and why reverse wins when there is one loss and many parameters.
