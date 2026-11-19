# 2. Gradients

By the end of this page you have a model that starts out knowing nothing and teaches itself the table that lesson 1 had to count.

Counting worked because the model was a table. For a model of any other shape there is nothing to count. We need a method that starts with wrong numbers and improves them.

## The idea

A model is a calculation with adjustable numbers in it, called parameters. Give the bigram one parameter per pair of characters, a score, and start them all at 0.

A score can be any number, so we need a rule that turns a row of scores into chances. Raise e to the power of each score, then divide by the row's total. The results are positive and sum to 1. The rule is called softmax. With all scores at 0, each character gets the same chance.

The rate at which the loss moves when one score rises a little is that score's slope. A slope of -2 means the loss falls by about 0.02 when the score rises by 0.01. The slopes for all the parameters, taken together, are the gradient.

Learning is one rule, repeated:

new value = old value - rate × slope

A positive slope means raising the parameter hurts, so we lower it. The rate, called the learning rate, sets the size of the step. Repeating this rule is called gradient descent.

One way to get a slope is to nudge. Add 0.000001 to one score, work out the loss again, and divide the change by 0.000001. That costs one full loss calculation per parameter, on each step. Our sentence has 10 different characters, so its model has 100 scores. GPT-2 has 124 million parameters.

One pass backwards gets all the slopes, because there is one loss. Going forwards would need one pass for each parameter. A loss is built from a few small operations, and each has a known slope:

| Operation | Slope for x |
|---|---|
| x + y | 1 |
| x × y | y, the other factor |
| x to the power k | k × x to the power k - 1 |
| e to the power x | e to the power x, the result itself |
| ln x | 1 / x |

Slopes combine by two rules, together called the chain rule. If x moves y at a rate of 3, and y moves the loss at a rate of 2, then x moves the loss at a rate of 6: multiply along a path. If x reaches the loss by two paths, add them.

We work out the loss once and let each number remember what it was made from. Then we walk backwards from the loss and work out each number's slope. Code that does this is called automatic differentiation, or autograd.

## By hand

Take L = a × b + c with a = 2, b = -3, c = 10. Then L = 4.

Name the middle step d = a × b = -6, so L = d + c.

- c moves L one for one. Its slope is 1. So is d's.
- a moves d at a rate of b, which is -3, and d moves L at a rate of 1. Along the path: -3.
- b moves d at a rate of a. Its slope is 2.

Check a by nudging. With a = 2.001, L = 2.001 × -3 + 10 = 3.997. L moved by -0.003 for a nudge of 0.001, a slope of -3.

Let the loss be (w - 3)². By the power rule and the chain rule its slope is 2 × (w - 3). Start at w = 0 with a rate of 0.1.

| Step | w | Slope | New w |
|---|---|---|---|
| 1 | 0 | -6 | 0.6 |
| 2 | 0.6 | -4.8 | 1.08 |
| 3 | 1.08 | -3.84 | 1.464 |

Each step closes a fifth of the gap to 3.

## The code

`class Value` defines a new kind of object. `__init__` runs when one is made, and `self` is the object itself. A `Value` holds a number, the Values it was made from, and how fast it moves when each of them moves:

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

Python calls `__add__` when it meets `a + b` and `__mul__` for `a * b`. So plain-looking arithmetic on Values builds new Values that remember their parents, with the slopes from the table. The `isinstance` line lets a plain number take part by wrapping it in a Value.

`backward` makes a list, `order`, of each Value after the Values it was made from. The result moves one for one with itself, so its `grad` is 1. Then `backward` walks the list in reverse. A Value must have collected the slopes from every Value made from it before it passes its own on:

```python
# gpt/autograd.py
        self.grad = 1.0
        for v in reversed(order):
            for parent, slope in zip(v.parents, v.slopes):
                parent.grad += slope * v.grad
```

One step of learning for the table of scores `W`:

```python
# gpt/autograd.py
        loss = bigram_loss(W, pairs)
        loss.backward()
        for row in W:
            for w in row:
                w.data -= rate * w.grad
                w.grad = 0.0
```

`bigram_loss` is lesson 1's loss with softmax in place of the counted table. The loop sets each `grad` back to 0 after use, or the next step's slopes would be added to this step's.

## The proof

```bash
uv run pytest tests/test_autograd.py
```

- Our gradients equal those of PyTorch, a library we start using in lesson 3, on an expression that uses each operation.
- Each operation's slope matches the table, and the chain rule agrees with nudging.
- The hand examples give -3, 2, 1, and the path 0.6, 1.08, 1.464.
- A number used twice collects both paths: in `a + a`, the slope for `a` is 2.
- A rate of 1.1 on the hand example runs away from 3.
- The learned table starts at ln 10 and after 300 steps is within 0.01 nats of the counted one.

```bash
uv run python -m gpt.autograd
```

```
counted table:       0.5152 nats
learned, step    0:  2.3026 nats
learned, step    1:  1.9895 nats
learned, step   10:  0.8852 nats
learned, step  100:  0.5404 nats
learned, step  299:  0.5231 nats
learned, step 2999:  0.5160 nats
one step on 2,000 pairs of Shakespeare builds 277,188 Values
```

After 3,000 steps the learned table scores 0.5160 nats, 0.0008 above the counted table's 0.5152.

## What it still cannot do

It learns one sentence. Each `Value` is a Python object holding one number, and one step on 2,000 pairs builds 277,188 of them. One step on all 1,003,853 pairs would build about 139 million.

## Check yourself

Predict before you run: after 3,000 steps the loss is 0.5160. Change `steps=3000` in the script to 30,000. Will the loss pass the counted 0.5152?

<details><summary>Answer</summary>

No. It keeps closing in and never arrives. No table beats the counted one, the maximum-likelihood table, and softmax reaches its chances of 0 and 1 only with scores of infinite size.
</details>

<details><summary>What is the slope of x³ at x = 2? Check by nudging.</summary>

3 × 2² = 12. With x = 2.001, x³ = 8.012006. The change, 0.012006, divided by 0.001 is about 12.
</details>

<details><summary>From lesson 1: where on this page does the loss of a blind guess over 10 characters appear?</summary>

Step 0. All scores are 0, softmax gives each character 1/10, and the surprise is ln 10 = 2.3026.
</details>

Go deeper:

- Andrej Karpathy, [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0). The same engine, built live.
- 3Blue1Brown, [Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus). The chain rule in pictures.
