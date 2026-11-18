# 9. The block

By the end of this page you have a unit whose gradients survive a stack of 24 as well as a stack of 2.

One attention layer is one round of reading, and we want several rounds. Attention also only mixes: with no bend, no token computes on what it read. And a plain stack of many such layers passes almost no slope back to the first.

## The idea

A block is attention with three parts added: an MLP, a path around each layer, and a normalization.

After attention, each token's vector passes through a small MLP: a linear layer that widens it four times, a bend, and a linear layer that narrows it back. The same MLP serves each token. Attention moves information between tokens, and the MLP computes on what each token gathered.

The bend is called GELU. For large negative numbers it gives about 0, and for large positive numbers about the number itself. Unlike tanh it does not flatten for large positive numbers, so slopes pass back through it there. ReLU, max(0, x), is the plainer choice. GELU is GPT-2's.

We also add each layer's result to its input. In place of `x = layer(x)` we write `x = x + layer(x)`. Each layer now makes a correction to the vector and hands the rest on unchanged. The line of plain additions that runs from the first block to the last is called the residual stream.

Lesson 2 showed that an addition passes a slope back unchanged. So each block has a path around it with a slope of exactly 1, and however many blocks we stack, the early ones still get slope from the loss. A test measures it with the starting weights of GPT-2, the 2019 model that lesson 10 rebuilds. Each block multiplies what passes through it by small weights (numbers with a spread of 0.02, which lesson 10 sets out). Without the additions, two blocks already pass back less than a hundredth of the gradient, and 24 blocks pass back less than a trillionth of it. With the additions, the gradient reaching the first block is about as large at 24 blocks as at 2, within a factor of 2.

The normalization keeps the numbers at a steady size. Before each layer, take each token's vector, subtract its mean and divide by its spread. The vector now has a mean of 0 and a spread of 1, whatever the blocks before it added. A learned gain and shift for each entry let the model move away from that when it helps. This is layer normalization. Steady sizes matter for the reason lesson 6 gave: scores that are too large put nearly all of a softmax's weight on one token.

We normalize the input to each layer and leave the residual stream alone:

`x = x + attention(norm(x))`, then `x = x + mlp(norm(x))`

The first model built from blocks like these, in 2017, normalized after each addition, which put a normalization on the main path. GPT-2 puts it where we have it.

## By hand

Normalize the vector (2, 4, 6, 8).

| Step | Result |
|---|---|
| Mean | 5 |
| Subtract it | (-3, -1, 1, 3) |
| Mean of the squares | (9 + 1 + 1 + 9) / 4 = 5 |
| Spread, its square root | 2.236 |
| Divide | (-1.342, -0.447, 0.447, 1.342) |

GELU at four points:

| In | -1 | 0 | 1 | 3 |
|---|---|---|---|---|
| Out | -0.159 | 0 | 0.841 | 2.996 |

A residual step: if x = (1, 2) and the layer returns (0.1, -0.2), the new x is (1.1, 1.8).

## The code

```python
# gpt/gpt2.py
class LayerNorm(nn.Module):
    """Lesson 9. Give each token's vector a mean of 0 and a spread of 1, then rescale."""

    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))
        self.shift = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        spread = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5).sqrt()
        return self.gain * x / spread + self.shift
```

`nn.Parameter` marks a tensor as something to learn. The `1e-5` keeps the division safe when a vector's entries are all equal.

```python
# gpt/gpt2.py
def gelu(x):
    """A smooth bend: about 0 for negative numbers, about x for positive ones."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
```

This is the formula GPT-2 used, and we need the same one to run its weights in lesson 10.

```python
# gpt/gpt2.py
class Block(nn.Module):
    """Lesson 9. Read from the other tokens, then think, each time adding to what is there."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.norm2 = LayerNorm(width)
        self.wide = nn.Linear(width, 4 * width)
        self.narrow = nn.Linear(4 * width, width)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.narrow(gelu(self.wide(self.norm2(x))))
```

## The proof

```bash
uv run pytest tests/test_gpt2.py
```

- The hand examples give the normalized vector and the four GELU values above.
- Our LayerNorm equals PyTorch's, with random gains and shifts.
- Our GELU equals PyTorch's tanh form.
- With GPT-2's starting weights and no additions, 2 stacked blocks pass back less than a hundredth of the gradient, and 24 blocks less than a trillionth. With the additions, 24 blocks pass back between half and twice what 2 blocks do.
- With PyTorch's own starting weights the additions still help: the gradient is more than 100 times larger with them.
- A block cannot read the future. Replace token 4 and tokens 0 to 3 do not change by a bit.

## What it still cannot do

Nothing yet turns the last block's vectors into scores for the next token, and nothing has been trained. We have the parts and no model.

Each test here checks one part against a small reference. We have no evidence yet that the parts add up to GPT-2.

## Check yourself

Predict before you run: how many parameters does one block have at a width of 128? Count the two normalizations, attention's two linear layers and the MLP's two.

<details><summary>Answer</summary>

198,272. Attention holds 49,536 + 16,512, the MLP holds 66,048 + 65,664, and the normalizations 256 each. The MLP is two thirds of the block.
</details>

<details><summary>What does layer normalization return for (5, 5, 5, 5)?</summary>

(0, 0, 0, 0), with the gain at 1 and the shift at 0, as they start. Subtracting the mean leaves zeros, and zero divided by a small positive number is zero.
</details>

<details><summary>The MLP uses the same weights for each token. How can it treat the third token unlike the first?</summary>

It reads the token's vector and nothing else. The vector already carries the place, added in lesson 8, and whatever attention gathered.
</details>

<details><summary>From lesson 2: why does `x + layer(x)` pass the slope back to x unchanged?</summary>

In a sum, each side moves the result one for one, so its slope is 1. The slope arriving at the sum is handed to x times 1, plus whatever comes back through the layer.
</details>

Go deeper:

- He and others, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2015). The addition, and what it did for depth.
- Ba, Kiros and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450) (2016).
- Xiong and others, [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (2020). Why normalizing before each layer needs less tuning than after.
