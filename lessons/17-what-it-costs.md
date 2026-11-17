# 17. What it costs

By the end of this page you can work out a model's memory and its arithmetic before you run it, and you have this Mac's speed to turn the arithmetic into hours.

We have not yet counted what the model costs: the memory a run needs and the multiplies a step takes.

## The idea

Start with memory. A parameter is one 32-bit number, 4 bytes. Our model of 795,392 parameters is 3.2 MB.

Training holds four numbers for each parameter: the weight, its slope, and AdamW's two averages. That is 16 bytes a parameter, 12.7 MB here.

Training also holds the result of each layer, because autograd needs them to walk backwards. These grow with the batch, the context, the width and the number of blocks, and they do not depend on the vocabulary. For one batch of 32 windows of 64 tokens they come to 54.3 MB, 17 times the weights. On a small model the weights are the small part.

One result grows faster than the rest. Attention makes a grid of scores with a row and a column for each token, so doubling the context makes that grid four times as large. At GPT-2's context of 1,024, one head's grid for one window is a million numbers. The grid does not have to exist all at once: the same result can be worked out a few rows at a time, and a test checks that.

Now the arithmetic. A linear layer multiplies each input by a weight and adds the products, so each weight costs one multiply and one add for each token. Sum that over the model's grids: 2 multiplies and adds for each weight in a grid, for each token. The lookup of a token's row costs none, and the scores layer uses that same grid once.

Attention adds work of its own. Each token scores each key and then mixes each value: 4 × context × width for each block. In our model that is 131,072 a token against 1,588,480 for the weights, 8% on top.

A training step runs forward and then backward. The run below times both: a step takes 3.0 times as long as the forward pass alone. So training costs about 6 multiplies and adds for each weight and each token.

Divide the work by the machine's speed and you have the time. This Mac's graphics chip trains a model of 5 million parameters at 307 billion multiplies and adds a second.

Writing text is a different shape of work. With lesson 16's cache, one new token reads each weight once and each cached key and value once. That is half a multiply-and-add for each byte read. A training step does 830 for each byte of its state. Writing does little arithmetic for each byte it reads, and training does a great deal. On large models that makes writing wait on memory and training wait on the chip. We have not measured which one limits this small model.

## By hand

SwiGLU's first grid in our model takes 128 numbers in and gives 341 out.

128 × 341 = 43,648 weights, so 87,296 multiplies and adds for each token.

The cache. Each block keeps a key and a value for each token, each with 4 heads of 32 numbers, at 4 bytes a number. There are 4 blocks.

2 × 4 × 4 × 32 × 4 = 4,096 bytes a token

With 2 key heads in place of 4 it is 2,048.

## The code

```python
# gpt/cost.py
def forward_flops(model, tokens, context):
    """Multiplies and adds in a forward pass over `tokens` tokens, read in windows of `context`."""
    grids = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    width, layers = model.embed.weight.size(1), len(model.blocks)
    weights = 2 * grids  # each token meets each weight once: one multiply and one add
    reading = 4 * context * width * layers  # each token scores every key, then mixes every value
    return (weights + reading) * tokens


def cache_bytes(model, tokens):
    """What lesson 16's cache holds for `tokens` tokens: a key and a value in each block."""
    attn = model.blocks[0].attn
    return 2 * len(model.blocks) * attn.k.out_features * 4 * tokens  # 4 bytes a number
```

`attn.k.out_features` is the number of key heads times the width of one head.

## The proof

```bash
uv run pytest tests/test_cost.py
```

- The formula equals a count. Hooks on each linear layer and on `attend` add up each multiply and add in a real forward pass, and the total is the formula's, to the last one.
- A forward pass costs between 1.0 and 1.2 times 2 for each parameter and token. The rest is attention's own work.
- After one step, the weights, slopes and two averages hold 16 bytes for each parameter.
- The cache formula equals the bytes in a real cache, with 4, 2 and 1 key heads.
- Layer results double when the batch doubles, and when the context doubles.
- Attention worked out in tiles of 1, 3 or 8 rows equals attention all at once.

```bash
uv run python -m gpt.cost
```

```
parameters: 795,392, 3.2 MB of weights
training state: weights, slopes and two averages, 12.7 MB
layer results for one batch of 32 x 64 tokens: 54.3 MB
forward pass over 2,048 tokens: 3.52 billion multiplies and adds
forward pass: 10 ms, 353 billion a second
training step: 30 ms, 3.0 times the forward pass
cache: 4,096 bytes a token
one new token after 64: 0.50 for each byte read
one training step: 830 for each byte of training state
on the graphics chip, 5,374,464 parameters: 986 ms a training step
that is 307 billion multiplies and adds a second
three hours at that speed: 3.3e+15 multiplies and adds
```

Timings are from an M4 laptop and move from run to run.

## What it still cannot do

A model of 5 million parameters would store Shakespeare, as lesson 13's smaller one did: a megabyte is too little text for it. We need more text, and real text arrives as a download.

## Check yourself

Predict before you run: GPT-2 has 124,439,808 parameters. How much memory do its weights, slopes and two averages take in training?

<details><summary>Answer</summary>

16 bytes each: 1,991,036,928 bytes, about 2 GB, before a single layer result.
</details>

<details><summary>A model has 1 million parameters in its grids. About how many multiplies and adds does it take to train on 1 million tokens?</summary>

About 6 for each weight and token: 6 × 10¹². At 307 billion a second that is 20 seconds.
</details>

<details><summary>The context doubles from 64 to 128. What happens to attention's grid of scores for one window?</summary>

It has twice the rows and twice the columns, so it is four times as large.
</details>

<details><summary>From lesson 16: why does the cache halve with 2 key heads in place of 4?</summary>

The cache holds keys and values, and grouped heads make half as many of each. The queries are not kept.
</details>

Go deeper:

- Austin and others, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). Counting memory, arithmetic and time for models a million times this size.
- Chen, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). The same sums for writing text, with the cache.
- Dao and others, [FlashAttention](https://arxiv.org/abs/2205.14135) (2022). Attention in tiles, done so that the grid of scores never leaves the chip's fastest memory.
