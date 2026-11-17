# 26. LoRA

By the end of this page the model learns lesson 25's skill again with 43,008 trained numbers in place of 5,374,464, the old weights do not move, and the new ones fold back in to leave a plain model.

Lesson 25's fine-tune moved each of the model's weights, and the result is a second file of 21.5 MB. Ten tasks would mean ten such files.

## The idea

Fine-tuning turns a grid of weights W into W plus a change D, and D is as large as W. LoRA bets that the change a new task needs is simpler than that, and builds D from two thin grids.

Take a grid that maps 256 numbers to 256. In place of a free D, write D as B times A. A maps the 256 numbers down to r of them, and B maps those r back up to 256. With r = 8, A and B hold 4,096 numbers together, where D held 65,536. The number r, the fewest numbers the change can be passed through, is called its rank.

What is lost is freedom. Whatever goes in, the change that comes out is a sum of B's 8 columns, each times one of the 8 numbers it passed through. A free D could move each of the 256 outputs on its own.

The layer becomes the old result plus the correction:

W x + scale × B (A x)

W is frozen: it gets no slope and takes no step. Only A and B train. The scale is a fixed number that sets how much the correction counts. Ours is 2, and we did not tune it. A starts random and B starts at zero, so the correction starts at zero and the model begins as the model it was.

We put a correction beside two grids in each attention: the one that makes queries and the one that makes values. That was the choice in the paper that introduced LoRA.

Two things can be done with a trained A and B. Keep them apart: one copy of the big model serves many tasks, each with a small file of its own. Or fold them in: add scale × B A to W once, and the model is a plain model again, no larger and no slower than before.

The run repeats lesson 25: the same examples, 2,000 steps of 32, the same three measures. Two settings differ. The peak rate is 0.002, twice lesson 25's, and there is no weight decay. We tried no others.

## By hand

The count for one block, at a rank of 8.

| Grid | Maps | A and B | Trained numbers |
|---|---|---|---|
| Queries | 256 to 256 | 8 × 256 and 256 × 8 | 4,096 |
| Values | 256 to 128 | 8 × 256 and 128 × 8 | 3,072 |

The values grid is the smaller because of lesson 16's grouped heads: our 8 heads of 32 share 4 sets of keys and values, and 4 × 32 = 128. A block has 7,168 trained numbers, and six blocks have 43,008. That is 0.8% of 5,374,464, and 0.17 MB at 4 bytes each.

A rank of 1, small enough to see. A is the row (1, 2) and B is the column (3, 0).

| Input | A x | B (A x) |
|---|---|---|
| (1, 0) | 1 | (3, 0) |
| (0, 1) | 2 | (6, 0) |
| (5, -1) | 3 | (9, 0) |

Each result is a multiple of (3, 0). No input can make this correction change the second number.

## The code

```python
# gpt/lora.py
class LoRA(nn.Module):
    """A frozen linear layer, plus the product of two thin grids that are trained."""

    def __init__(self, layer, rank, scale=2.0):
        super().__init__()
        self.layer, self.scale = layer, scale
        self.down = nn.Parameter(torch.randn(rank, layer.in_features) / rank)  # width to rank
        self.up = nn.Parameter(torch.zeros(layer.out_features, rank))  # zero: no change at first

    def forward(self, x):
        return self.layer(x) + self.scale * (x @ self.down.T @ self.up.T)


GRIDS = ("q", "k", "v", "proj", "gate", "wide", "narrow")  # each linear layer in a block


def add_lora(model, rank, names=("q", "v")):
    """Freeze the whole model, then put a correction beside each grid in a block that has one of
    these names. The two named here make the queries and the values."""
    model.requires_grad_(False)
    for block in model.blocks:
        for part in (block.attn, block.mlp):
            for name, layer in list(part.named_children()):
                if name in names:
                    setattr(part, name, LoRA(layer, rank))


@torch.no_grad()
def merge(model):
    """Add each correction into its grid and put the plain layer back. The model computes the same."""
    for block in model.blocks:
        for part in (block.attn, block.mlp):
            for name, one in list(part.named_children()):
                if isinstance(one, LoRA):
                    one.layer.weight += one.scale * one.up @ one.down
                    setattr(part, name, one.layer)
```

`down` is A and `up` is B, and `in_features` and `out_features` are a layer's counts of inputs and outputs. `x` arrives as a row, so B (A x) is written `x @ down.T @ up.T`. Dividing by the rank keeps A's first numbers small. `requires_grad_(False)` tells PyTorch to keep no slopes for the model's own weights. `nn.Parameter` marks the two new grids as weights that do train. `getattr` and `setattr` read and replace a part of the model by its name, and `named_children` lists the parts, so nothing in the model's own file changes. PyTorch stores a layer's grid as outputs by inputs (lesson 4), the shape of `up @ down`.

## The proof

```bash
uv run pytest tests/test_lora.py
```

- Adding the corrections changes no score, to the last bit, until they are trained.
- Only grids named `down` and `up` train. In a small model they hold under a tenth of the weights.
- Training lowers the loss, moves `up` away from zero, and leaves each frozen weight equal to what it was, to the last bit.
- After the merge the scores match to 0.00001, the model's parts have their old names, and no LoRA layer is left.
- With r = 3, PyTorch counts the rank of `up` times `down` as 3.

```bash
uv run python -m gpt.lora
```

```
@@OUTPUT@@
```

@@RESULTS@@

## What it still cannot do

Both fine-tunes learned from 60,404 stories that already obeyed their orders: a right answer to copy, for each example. For most things we want from a model, nobody has written the right answers. We have only a check that says pass or fail.

## Check yourself

Predict before you run: the corrections hold 0.8% of the weights. Will the model learn the task as well as lesson 25's did?

<details><summary>Answer</summary>

@@PREDICT@@
</details>

<details><summary>Why does B start at zero and A at random numbers, and not both at zero?</summary>

With B at zero the correction is zero, so the model starts unchanged. B's slope is multiplied by A x, and A's is passed back through B. With both at zero, both slopes are zero and neither grid ever moves. With A random, B moves on the first step, and A once B has.
</details>

<details><summary>A grid maps 512 numbers to 512. How many numbers does a correction of rank 4 train, against the grid's own?</summary>

4 × 512 + 512 × 4 = 4,096, against 262,144. That is 64 times fewer.
</details>

<details><summary>From lesson 17: training keeps four numbers for each weight it trains. What does freezing the model save?</summary>

The slope and AdamW's two averages for each frozen weight: 12 of the 16 bytes. Only A and B need all four.
</details>

Go deeper:

- Hu and others, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021). The idea, the choice of queries and values, and ranks as low as 1.
- Schulman and others, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (2025). When LoRA matches full fine-tuning, and which grids and rates it takes.
- Hugging Face, [PEFT](https://github.com/huggingface/peft). LoRA and its relatives as a library.
