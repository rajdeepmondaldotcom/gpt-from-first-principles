# 3. Tensors

By the end of this page the bigram learns all of Shakespeare in two seconds.

Lesson 2's model learned one sentence. Each number was a Python object, and a step on 2,000 pairs of Shakespeare took up to a second.

## The idea

Store the numbers in a grid and do arithmetic on the whole grid at once. The grid is called a tensor, and its shape says how many numbers lie along each direction. Our table of scores `W` has shape 65 × 65, one row per character. One instruction, such as `W.exp()`, runs over all 4,225 numbers in compiled code.

PyTorch gives us tensors, and an autograd like ours comes built in. On this page we still work the slopes out by hand, and use PyTorch's autograd only to check them.

Three tensor moves do the work.

The first is looking up many rows at once. Turn the text into a tensor of whole numbers, each character's place in the vocabulary. If `x` holds the first character of each pair, `W[x]` has shape 1,003,853 × 65: for each pair, the row of scores it needs.

The second is working along one direction. `scores.sum(dim=-1)` adds up each row. `dim=-1` means the last direction, across the 65 scores.

The third is stretching. `scores.max(dim=-1, keepdim=True).values` is a column holding each row's largest score. Subtract it from `scores` and PyTorch stretches the column across all 65 places. This is called broadcasting.

Now the slopes. For one pair, the loss is minus the log of the chance that softmax gave to the true next character. Work through the algebra and the slope for each score comes out short:

- for the true next character: its chance, minus 1
- for each other character: its chance

Raising any score makes the row's total bigger, which costs each character some chance. Raising the true character's own score also wins that back and more. The slopes in a row sum to zero. The loss is an average over N pairs, so each slope is divided by N.

A lookup sends slopes back to the row it read. A row read by 500 pairs collects 500 sets of slopes. That is "add across paths" from lesson 2.

Big scores break the arithmetic. A computer stores a number in a fixed count of bits, and e to the power 1,000 does not fit. PyTorch writes `inf`, and `inf / inf` gives `nan`, which means "not a number". Softmax depends only on the gaps between scores: add 5 to each score in a row and the chances stay put. So we subtract the row's largest score first. The largest becomes 0, e to the 0 is 1, and nothing overflows.

One idea is left. A step on all 1,003,853 pairs takes about a second. A step on 2,000 pairs picked at random takes a few milliseconds, and its slopes point almost the same way, with some noise. The random handful is called a batch. Learning from batches is called stochastic gradient descent. Two thousand cheap, noisy steps get further than a few exact ones.

## By hand

One row of scores, (2, 1, 0). The true next character is the first.

| | First | Second | Third |
|---|---|---|---|
| Score | 2 | 1 | 0 |
| e to the score | 7.389 | 2.718 | 1 |
| Chance (divide by 11.107) | 0.665 | 0.245 | 0.090 |
| Slope | 0.665 - 1 = -0.335 | 0.245 | 0.090 |

The loss is -ln 0.665 = 0.408. The slopes sum to zero.

Now the row (1000, 999, 998). Subtract the largest: (0, -1, -2). e to those powers gives 1, 0.368 and 0.135, which total 1.503. The chances are 0.665, 0.245 and 0.090 again, because the gaps between the scores did not change.

## The code

```python
# gpt/tensors.py
def softmax(scores):
    """Lesson 2's softmax for a grid: each row of scores becomes a row of chances."""
    scores = scores - scores.max(dim=-1, keepdim=True).values  # same chances, no overflow
    exps = scores.exp()
    return exps / exps.sum(dim=-1, keepdim=True)
```

```python
# gpt/tensors.py
def cross_entropy_slopes(scores, targets):
    """The slope of that loss for each score, worked out by hand."""
    slopes = softmax(scores)
    slopes[torch.arange(len(targets)), targets] -= 1
    return slopes / len(targets)
```

`slopes[torch.arange(len(targets)), targets]` picks one number from each row: in row 0 the place `targets[0]`, in row 1 the place `targets[1]`.

One step of learning:

```python
# gpt/tensors.py
        x, y = ids[at], ids[at + 1]
        slopes = cross_entropy_slopes(W[x], y)  # one row of slopes per pair
        # Each pair read row x of W, so its slopes are added to that row.
        W -= rate * torch.zeros_like(W).index_add_(0, x, slopes)
```

`at` holds the positions of the pairs in this batch. `index_add_(0, x, slopes)` adds row i of `slopes` into row `x[i]` of a grid of zeros.

## The proof

```bash
uv run pytest tests/test_tensors.py
```

- The hand example gives the chances, the loss and the slopes above.
- Our softmax and cross-entropy equal PyTorch's.
- The slopes worked out by hand equal PyTorch's autograd.
- Scores of 1,000 give `nan` the naive way and the right chances our way.
- A lookup sends its slopes back to the rows it read, and a row nobody read gets none.
- On the sentence, the tensor model gives the same loss as lesson 2's model after 1, 10 and 30 steps.
- Random batches end within 0.01 nats of using the whole text.

```bash
uv run python -m gpt.tensors
```

```
one step on 2,000 pairs:     4.8 ms
one step on 1,003,853 pairs: 1.38 s
2,000 steps on batches of 2,000 pairs: 2 s
learned bigram, training text: 2.465 nats
learned bigram, unseen text:   3.599 bits per byte
```

Timings are from an M4 laptop and move from run to run. A step on 2,000 pairs is about a hundred times faster than in lesson 2.

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| Bigram, learned | 3.599 |

The learned table sits 0.018 bits behind the counted one. Its steps are noisy, and the rows for rare characters get few of them.

## What it still cannot do

```
Once upon a timer dle PEved himalef by rol y t, han, it
```

It is the same model as lesson 1, learned by a different route, so it writes the same kind of soup. It still sees one character back.

We can now train a model of any shape. The table is the wrong shape: more context means a row for each possible run of characters, and almost all of those rows stay empty.

## Check yourself

Predict before you run: `W[x]` for the training text holds 1,003,853 × 65 numbers at 4 bytes each. How much memory is that?

<details><summary>Answer</summary>

About 261 MB. That is why a step on all the pairs is slow.
</details>

<details><summary>A row of scores is (0, 0, 0, 0) and the true next character is the second. What are the slopes?</summary>

Each chance is 1/4. The slopes are 0.25, -0.75, 0.25, 0.25.
</details>

<details><summary>Why does adding 5 to each score in a row leave the chances unchanged?</summary>

Each e-to-the-score is multiplied by e to the 5, and so is their total. The factor cancels in the division.
</details>

<details><summary>Tensor `a` has shape 3 × 1 and tensor `b` has shape 3 × 4. What is the shape of `b - a`, and what was subtracted from row 2?</summary>

3 × 4. The one number in row 2 of `a` was subtracted from all four numbers in row 2 of `b`.
</details>

<details><summary>From lesson 2: a row was read by three pairs in a batch. Which rule decides what slope it gets?</summary>

Add across paths. The row reaches the loss three times, so the three sets of slopes add.
</details>

Go deeper:

- Andrej Karpathy, [Becoming a backprop ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI). Slopes by hand through a whole network.
- PyTorch, [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html). The exact stretching rules.
- Blanchard, Higham and Higham, [Accurate computation of the log-sum-exp and softmax functions](https://arxiv.org/abs/1909.03469). Why subtracting the largest score is safe.
