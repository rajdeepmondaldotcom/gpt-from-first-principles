# 26. LoRA

By the end of this page 43,008 trained numbers, in place of 5,374,464, have learned part of lesson 25's skill. The old weights have not moved, and the new ones fold back in to leave a plain model.

Lesson 25's fine-tune moved all 5,374,464 weights. The result is a second file of 21.5 MB, and plain stories got 0.069 nats worse.

## The idea

Fine-tuning turns a grid of weights W into W plus a change D, and D is as large as W. The authors of LoRA, low-rank adaptation, bet that a new task needs a simpler change, and build D from two thin grids.

Take a grid that maps 256 numbers to 256. In place of a free D, write D as B times A. A maps the 256 numbers down to r of them, and B maps those r back up to 256. With r = 8, A and B hold 4,096 numbers, where D held 65,536. The number r, the fewest numbers the change can be passed through, is called its rank.

The layer becomes the old result plus the correction:

W x + (32 / r) × B (A x)

W is frozen: it gets no slope and takes no step. Only A and B train.

The number in front sets how much the correction counts. We divide it by r so that the first step is the same on average at each rank, and one learning rate can serve them all.

A starts random and B at zero, so the correction starts at zero and the model begins as it was.

A block has seven grids: four in attention and three in the MLP. The paper that introduced LoRA put a correction beside two, the ones that make queries and values. We run that first, at a rank of 8. Then, one change at a time as lesson 14 taught: eight times the rank, and a correction beside each of the seven.

Keep a trained A and B apart, and one copy of the big model can serve many tasks, each with a small file of its own. Or fold them in: add (32 / r) × B A to W once, and the model is plain again, no larger and no slower.

Each run repeats lesson 25: the same examples in the same batches, 2,000 steps of 32, the same three measures. Two settings differ: the peak rate is 0.01, ten times as high, and there is no weight decay. We took the 32, the division by r and the ten times from Schulman and others, below, without a search.

## By hand

The count for one block, at a rank of 8.

| Grid | Maps | A and B | Trained numbers |
|---|---|---|---|
| Queries | 256 to 256 | 8 × 256 and 256 × 8 | 4,096 |
| Values | 256 to 128 | 8 × 256 and 128 × 8 | 3,072 |

The values grid is smaller because of lesson 16's grouped heads: 8 heads of 32 share 4 sets of keys and values, and 4 × 32 = 128. A block has 7,168 trained numbers and six blocks 43,008: 0.8% of 5,374,464, and 0.17 MB. At a rank of 64 the same two grids have 344,064, or 6.4%. A rank of 8 beside each of the seven grids has 221,088, or 4.1%.

A rank of 1, small enough to see. A is the row (1, 2) and B is the column (3, 0).

| Input | A x | B (A x) |
|---|---|---|
| (1, 0) | 1 | (3, 0) |
| (0, 1) | 2 | (6, 0) |
| (5, -1) | 3 | (9, 0) |

Each result is a multiple of (3, 0). No input can make this correction change the second number, and a free D could.

## The code

```python
# gpt/lora.py
class LoRA(nn.Module):
    """A frozen linear layer, plus the product of two thin grids that are trained."""

    def __init__(self, layer, rank):
        super().__init__()
        self.layer = layer
        self.scale = 32 / rank  # over the rank: the first step is the same size on average
        edge = layer.in_features**-0.5
        self.down = nn.Parameter(torch.empty(rank, layer.in_features).uniform_(-edge, edge))
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

`down` is A and `up` is B. `x` arrives as a row, so B (A x) is written `x @ down.T @ up.T`. `requires_grad_(False)` tells PyTorch to keep no slopes for the model's own weights. `named_children` lists a part's layers by name and `setattr` replaces one, so the model's own file does not change.

## The proof

```bash
uv run pytest tests/test_lora.py
```

- Adding the corrections changes no score, to the last bit, until they are trained.
- Only `down` and `up` train. In a small model they hold under a tenth of the weights.
- Training lowers the loss, moves `up` away from zero, and leaves each frozen weight as it was, to the last bit.
- After the merge the scores match to 0.00001, the model's parts have their old names, and no LoRA layer is left.
- Beside each of the seven grids: no change at first, the loss falls, and the merge leaves a plain model.
- After training at r = 3 and the merge, the change in a grid has a rank of 3.

```bash
uv run python -m gpt.lora
```

```
before: 0 of 200 tasks passed, 95% interval 0.00 to 0.02
before: 0.14 of the asked words used, 95% interval 0.11 to 0.17
before: 177 stories end by themselves, plain stories 1.480 nats a token
rank 8, queries and values: 43,008 of 5,374,464 parameters train, 0.17 MB to store
step 500: 1.289 nats on the answers
step 1000: 1.302 nats on the answers
step 1500: 1.292 nats on the answers
step 2000: 1.233 nats on the answers
rank 8, queries and values: 10 of 200 tasks passed, 95% interval 0.02 to 0.09
rank 8, queries and values: 0.37 of the asked words used, 95% interval 0.33 to 0.41
rank 8, queries and values: 190 stories end by themselves, plain stories 1.487 nats a token
rank 8, queries and values: plain stories, after minus before, +0.006 nats, 95% interval +0.005 to +0.008
rank 64, queries and values: 344,064 of 5,374,464 parameters train, 1.38 MB to store
step 500: 1.280 nats on the answers
step 1000: 1.295 nats on the answers
step 1500: 1.282 nats on the answers
step 2000: 1.218 nats on the answers
rank 64, queries and values: 12 of 200 tasks passed, 95% interval 0.03 to 0.10
rank 64, queries and values: 0.41 of the asked words used, 95% interval 0.37 to 0.44
rank 64, queries and values: 192 stories end by themselves, plain stories 1.489 nats a token
rank 64, queries and values: plain stories, after minus before, +0.009 nats, 95% interval +0.007 to +0.011
rank 8, each grid in a block: 221,088 of 5,374,464 parameters train, 0.88 MB to store
step 500: 1.341 nats on the answers
step 1000: 1.332 nats on the answers
step 1500: 1.295 nats on the answers
step 2000: 1.225 nats on the answers
rank 8, each grid in a block: 7 of 200 tasks passed, 95% interval 0.01 to 0.07
rank 8, each grid in a block: 0.34 of the asked words used, 95% interval 0.29 to 0.37
rank 8, each grid in a block: 194 stories end by themselves, plain stories 1.490 nats a token
rank 8, each grid in a block: plain stories, after minus before, +0.009 nats, 95% interval +0.007 to +0.012
asked words used, rank 64, queries and values minus rank 8, queries and values: +0.037, 95% interval -0.005 to +0.080
asked words used, rank 8, each grid in a block minus rank 8, queries and values: -0.033, 95% interval -0.080 to +0.012
merged against unmerged, largest difference in a score: 0.000014
```

Beside queries and values at a rank of 8, the model uses 0.37 of the asked words: 0.33 to 0.41. That is clear of the 0.14 it started from and of the full fine-tune's 0.56, so the corrections learned part of the skill. It passes 10 of 200 tasks, against the 27 of lesson 25's draw.

Eight times the rank gives 0.41: a gap of +0.037, interval -0.005 to +0.080. A correction beside each grid gives 0.34: -0.033, interval -0.080 to +0.012. Neither is clear of zero, so these runs do not rank the settings. None reaches 0.56.

Plain stories pay less too: 0.006 to 0.009 nats, against 0.069.

Folded in, the last run's corrections move no score on a 64-token window by more than 0.000014.

## What it still cannot do

Each fine-tune so far learned from 60,404 stories that obeyed their orders, and the best still leaves out 0.44 of the asked words. None trains on the misses the check finds.

## Check yourself

Predict before you run: beside queries and values, the corrections hold 0.8% of the weights. Will the model learn the task as well as lesson 25's did?

<details><summary>Answer</summary>

Not in these 2,000 steps: 0.37 of the asked words against 0.56, and 10 tasks against 27.
</details>

<details><summary>Why does B start at zero and A at random numbers, and not both at zero?</summary>

With B at zero the correction is zero, so the model starts unchanged. B's slope is multiplied by A x, and A's is passed back through B. With both at zero, neither grid ever moves. With A random, B moves on the first step, and A once B has.
</details>

<details><summary>From lesson 17: training keeps four numbers for each weight it trains. What does freezing the model save?</summary>

The slope and AdamW's two averages for each frozen weight: 12 of the 16 bytes. Only A and B need all four.
</details>

Go deeper:

- Hu and others, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021). The idea, the choice of queries and values, and ranks as low as 1.
- Schulman and others, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (2025). When LoRA matches full fine-tuning, and which grids and rates it takes.
