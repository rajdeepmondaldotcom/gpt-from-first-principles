# 43. Split the model

By the end of this page no worker holds the whole model: you have shared out one layer's weights and the optimizer's averages among four workers, and checked that each sum comes out as it did in one piece.

We served a model of 7.6 billion parameters from one card. We could not train it there: lesson 17's 16 bytes a parameter come to 122 GB, and the card has 48. Lessons 20 and 41 gave each worker a whole copy, so more cards would not help.

## The idea

In training a worker holds four numbers for each parameter: the weight, its slope, and AdamW's two averages. With whole copies, each worker holds all four for the whole model. Two things can be shared out.

The averages first. AdamW steps each parameter from its own slope and its own two averages (lesson 11). No parameter's step reads another's. So one worker can keep the averages for a share of the tensors and step only those. It needs the summed slopes for its share, and the others need the weights it stepped. Rajbhandari and others call this ZeRO. They go on to share out the slopes and the weights too, and PyTorch ships the idea as FSDP.

Then a layer itself. Lesson 16's MLP widens a vector to hidden units and narrows it again. Each hidden unit reads the whole input through its own row of the two grids that widen, and adds its own column of the grid that narrows into the result. No unit reads another. So deal the units out. A worker holds its units' rows and columns, takes the whole input, and works out its part of the result. The parts add up to the result. Between cards that sum is an exchange, lesson 41's `all_reduce`, once for the MLP.

Backwards, the result was a sum, and lesson 2 showed that an addition passes a slope back unchanged. So each worker is handed the same slope, and from it finds the slopes of its own weights with no help. Each also finds a slope for the input, and the input's slope is the sum of those: a second exchange. Shoeybi and others split each MLP of a transformer this way, and its attention by heads.

Splitting changes what travels. Whole copies exchange slopes once a step. A split MLP exchanges a vector for each token, forwards and back, in each block.

## By hand

Lesson 41's story model has 14,555,904 parameters, a width of 384 and 8 blocks, and a step reads 32 windows of 256 tokens: 8,192 tokens.

| What each worker hands over in a step | The sum | Bytes |
|---|---|---|
| Its slopes | 14,555,904 × 4 | 58.2 MB |
| Its parts of 8 split MLPs, and the slopes back | 2 × 8,192 × 384 × 4 × 8 | 201.3 MB |

For this model the split moves over three times as much. Now take a made-up model of 70 billion parameters, 8,192 wide and 80 blocks deep, in 16-bit numbers. Its slopes are 140 GB. Its split MLPs move 2 × 8,192 × 8,192 × 2 × 80 bytes, 21.5 GB. Slopes grow with the parameters, and a split layer's traffic with the tokens and the width.

## The code

```python
# gpt/split.py
def deal(mlp, workers):
    """What each worker holds of lesson 16's MLP: a share of its hidden units, as their rows of
    the two grids that widen and their columns of the grid that narrows."""
    units = shares(mlp.narrow.in_features, workers)
    return [(mlp.gate.weight[u], mlp.wide.weight[u], mlp.narrow.weight[:, u]) for u in units]


def part(held, x):
    """One worker's part of the MLP's result: the whole input, through the weights it holds.
    The parts of all the workers add up to the result, and that sum is the one exchange."""
    gate, wide, narrow = held
    opened = x @ gate.T
    return (opened * torch.sigmoid(opened) * (x @ wide.T)) @ narrow.T


def dealt(params, workers, **how):
    """AdamW's two averages, shared out: worker i keeps them for each `workers`-th tensor from
    tensor i on, and steps only those. Returns one of lesson 11's optimizers for each worker."""
    return [AdamW(params[i::workers], **how) for i in range(workers)]
```

`shares` cuts a count into runs of nearly equal length. PyTorch stores a grid with a row for each output, so a hidden unit is a row of `gate` and `wide` and a column of `narrow`. `part` is lesson 16's MLP on a share of the units. `params[i::workers]` is every `workers`-th tensor from tensor `i` on, and `**how` passes the rate on to lesson 11's `AdamW`.

## The proof

```bash
uv run pytest tests/test_split.py
```

- The shares cover each unit once, and their lengths differ by one at most.
- With 1, 2, 3 or 5 workers the parts add up to the whole MLP's result. With one worker left out they do not.
- Each worker's slopes for its own weights equal the whole MLP's, and the input's slope is the sum of the workers' and not any one of them.
- Four steps with the averages shared among three end where one optimizer ends. With one worker that never steps they do not.
- A worker holds 16 bytes a parameter with whole averages, as lesson 17 counted.

Predict before you run: the workers' parts are added in another order than the whole MLP's one multiply. The shared-out AdamW does each parameter's sums in the same order as before. Which of the two will match the whole to the last bit?

```bash
uv run python -m gpt.split
```

```
lesson 41's story model: 14,555,904 parameters, shared among 4 workers
a whole copy for each worker: 232.9 MB each. With the averages shared out, the worker with the most holds 155.0 MB
one MLP of 1,179,648 weights: each worker holds 294,912. The parts add up to the whole result to within 0.000000112
each worker's slopes for its own weights match the whole MLP's to within 0.000000000, and the four slopes for the input add up to the whole one to within 0.000000125
three steps of AdamW with the averages shared among 4: the workers keep averages for 4,818,816, 3,245,952, 3,245,568, 3,245,568 parameters, and each weight ends equal to one optimizer's, to the last bit: True
what each worker hands over in a step of 8,192 tokens: its slopes, 58.2 MB (lessons 20 and 41). With each block's MLP split, a part forwards and a slope back for each token: 201.3 MB
```

Each worker holds a quarter of the MLP's weights, and the four parts give its result to seven decimal places. The slopes come out the same.

With the averages shared out, the fullest worker holds 155.0 MB where a whole copy is 232.9. The shares are uneven because we dealt whole tensors in turn, and the first worker drew the token rows, the largest tensor. An even deal would leave each with 145.6 MB: every weight and slope, and a quarter of the averages. That is still far from a quarter of 232.9, which is why ZeRO goes on to share out the slopes and the weights.

## What it still cannot do

These workers are four entries of a list in one process. Nothing crossed a wire and nothing was timed: the page proves the sums, and lesson 41's `all_reduce` would carry them between cards. We split one MLP and not the attention beside it, and never put different blocks on different cards. [MAP.md](../MAP.md) says where to read on.

Lesson 41's chess model is rated +775 on a scale of our own, where a random mover is 0. We do not know what that is worth against a player the world rates, or how strong this code can get.

## Check yourself

<details><summary>Prediction</summary>

The optimizer: its weights are equal to the last bit. The MLP's parts agree with the whole to about seven decimal places, because adding in another order rounds differently.
</details>

<details><summary>From lesson 17: the 7.6 billion parameter model trains on four cards with the averages shared evenly. Does each card's share fit in 48 GB?</summary>

No. Each holds every weight and slope, 8 × 7.6 = 60.8 GB, and a quarter of the averages, 15.2 GB: 76 GB. The weights and slopes have to be shared out too.
</details>

<details><summary>The MLP has three grids, all split. Why is one sum of the parts enough going forwards?</summary>

A worker's hidden units are used only by that worker's own columns of the narrowing grid. Nothing has to cross until the parts are added into the result.
</details>

Go deeper:

- Shoeybi and others, [Megatron-LM](https://arxiv.org/abs/1909.08053) (2019). An MLP's first grid split one way and its second the other, so that one exchange forwards and one back are enough.
- Rajbhandari and others, [ZeRO](https://arxiv.org/abs/1910.02054) (2019). The optimizer's averages shared out among the workers, then the slopes, then the weights.
- PyTorch, [FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html). PyTorch's own wrapper for sharing a model's parameters out.
