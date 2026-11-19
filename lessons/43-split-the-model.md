# 43. Split the model

By the end of this page no worker holds the whole model: you have shared out one layer's weights and the optimizer's averages among four workers, and checked that each sum comes out as it did in one piece.

We served a model of 7.6 billion parameters from one card. We could not train it there: lesson 17's 16 bytes a parameter come to 122 GB, and NVIDIA lists the card at 48. Lessons 20 and 41 gave each worker a whole copy, so more cards would not help.

## The idea

In training a worker holds four numbers for each parameter: the weight, its slope, and AdamW's two averages. With whole copies, each worker holds all four for the whole model.

We can share the averages out, because AdamW steps each parameter from its own slope and its own two averages (lesson 11). So one worker can keep the averages for a share of the model's tensors and step only those. It needs the summed slopes for its share, and the others need the weights it stepped: together, the same bytes as one `all_reduce` of the slopes. Rajbhandari and others call this the first stage of ZeRO. They go on to share out the slopes and the weights too, and PyTorch ships the idea as FSDP. Sharing out the weights means gathering each block's weights before its use, forwards and again backwards: more bytes than one `all_reduce` of the slopes.

A layer can be shared out as well. Lesson 16's MLP widens a vector to hidden units and narrows it again. Each hidden unit reads the whole input through its own row of the two grids that widen (PyTorch stores a row for each output). It then adds its own column of the grid that narrows, times its value, into the result. No unit reads another, so deal the units out. A worker holds its units' rows and columns, takes the whole input, and works out its part of the result. The parts add up to the result. Between cards that sum is an exchange, lesson 41's `all_reduce`, once for the MLP.

Backwards, the result was a sum, and lesson 2 showed that an addition passes a slope back unchanged. So each worker gets the same slope, and from it finds the slopes of its own weights with no help. Each also finds a slope for the input, and the input's slope is their sum, by lesson 2's rule for several paths: a second exchange. Shoeybi and others split each MLP of a transformer this way, and its attention by heads.

Whole copies add up slopes, once a step. A split MLP adds up a vector for each token, forwards and back, in each block. The timing differs: PyTorch's own version (lesson 20) sends slopes while the pass back still runs, but no block can start until its MLP's sum arrives.

## By hand

The split, small. Two hidden units with values h1 and h2, and their columns c1 and c2 of the narrowing grid, give the result h1·c1 + h2·c2. Worker 1 holds unit 1 and sends h1·c1, worker 2 sends h2·c2, and one sum gives the result.

Lesson 41's story model has 14,555,904 parameters, a width of 384, 8 blocks and MLPs of 1,024 hidden units (8 × 384 / 3, lesson 16). A step reads 32 windows of 256 tokens: 8,192 tokens.

| What the workers add up in a step | The sum | Bytes |
|---|---|---|
| Their slopes | 14,555,904 × 4 | 58.2 MB |
| Their parts of 8 split MLPs, and the slopes back | 2 × 8,192 × 384 × 4 × 8 | 201.3 MB |

The second row is forwards and back, for each token, each of 384 numbers at 4 bytes, in 8 blocks.

For this model the split adds up over three times as much. Now take a made-up model of 70 billion parameters, 8,192 wide and 80 blocks deep, in 16-bit numbers. Its slopes are 140 GB. With the same 8,192 tokens a step, its split MLPs add up 2 × 8,192 × 8,192 × 2 × 80 bytes, 21.5 GB. Slopes grow with the parameters, and a split layer's sums with the tokens and the width. Each block waits on its sum, so Shoeybi and others split a layer only among the cards of one server.

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

`narrow.in_features` is the count of hidden units, 1,024. `shares` cuts a count into runs of nearly equal length. `params[i::workers]` is every `workers`-th tensor from tensor `i` on, and `**how` passes the rate on to lesson 11's `AdamW`.

## The proof

```bash
uv run pytest tests/test_split.py
```

- The shares cover each unit once, and their lengths differ by one at most.
- With 1, 2, 3 or 5 workers the parts add up to the whole MLP's result. With one worker left out they do not.
- Each worker's slopes for its own weights equal the whole MLP's, and the input's slope is the sum of the workers'.
- Four steps with the averages shared among three end where one optimizer ends. With one worker that never steps they do not.

Predict before you run: the script checks the workers' MLP parts, and the weights after AdamW with its averages shared out, against the model in one piece. Which will match to the last bit?

```bash
uv run python -m gpt.split
```

```
lesson 41's story model: 14,555,904 parameters, shared among 4 workers
a whole copy for each worker: 232.9 MB each. With the averages shared out, the worker with the most holds 155.0 MB
one MLP of 1,179,648 weights: each worker holds 294,912. The parts add up to the whole result to within 0.000000112
each worker's slopes for its own weights match the whole MLP's to within 0.000000000, and the four slopes for the input add up to the whole one to within 0.000000125
three steps of AdamW with the averages shared among 4: the workers keep averages for 4,818,816, 3,245,952, 3,245,568, 3,245,568 parameters, and each weight ends equal to one optimizer's, to the last bit: True
what the workers add up between them in a step of 8,192 tokens: their slopes, 58.2 MB (lessons 20 and 41). With each block's MLP split, a part forwards and a slope back for each token: 201.3 MB
```

The four parts add up to the MLP's result to within 0.0000002, and so does the input's slope. Each worker's own slopes match to nine decimal places.

With the averages shared out, the fullest worker holds 155.0 MB where a whole copy is 232.9: we dealt whole tensors in turn, and the first drew the token rows, the largest. An even deal leaves 145.6 MB, every weight and slope and a quarter of the averages. That is still far from a quarter of 232.9.

## What it still cannot do

These workers are four entries of a list in one process. We sent nothing over a wire and timed nothing: the page proves the sums that lesson 41's `all_reduce` would carry between cards. Our steps did not clip. Lesson 12's clipping needs the whole slope's length, so each worker would first send its sum of squares. We split one MLP and not the attention beside it, and never put different blocks on different cards. [MAP.md](../MAP.md) says where to read on.

Lesson 41's chess model is rated +775 on a scale of our own, where a random mover is 0. We do not know what that is worth on a scale others use, or how strong this code can get.

## Check yourself

<details><summary>Prediction</summary>

The optimizer: it does the same sums in the same order, and its weights are equal to the last bit. The MLP's parts agree with the whole to within 0.0000002, because adding in another order rounds differently.
</details>

<details><summary>From lesson 17: the 7.6 billion parameter model trains on four cards with the averages shared evenly. Does each card's share fit in 48 GB?</summary>

No. Each holds every weight and slope, 8 × 7.6 = 60.8 GB, and a quarter of the averages, 15.2 GB: 76 GB. Split every grid four ways as well and each card holds 121.6 / 4 = 30.4 GB, before what the step keeps for the backward pass.
</details>

<details><summary>The MLP has three grids, all split. Why is one sum of the parts enough going forwards?</summary>

Only a worker's own columns of the narrowing grid use its hidden units. Nothing has to cross until the parts are added into the result.
</details>

Go deeper:

- Shoeybi and others, [Megatron-LM](https://arxiv.org/abs/1909.08053) (2019). An MLP split so that one exchange forwards and one back are enough, among cards of one server.
- Rajbhandari and others, [ZeRO](https://arxiv.org/abs/1910.02054) (2019). The optimizer's averages shared out among the workers, then the slopes, then the weights.
- PyTorch, [FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html). PyTorch's own version.
