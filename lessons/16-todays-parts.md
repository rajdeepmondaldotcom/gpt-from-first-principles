# 16. Today's parts

By the end of this page the model has the parts of Llama, and loaded with a tiny reference Llama's weights it gives that model's scores.

Lesson 15's model still has three parts from 2019: layer normalization, the GELU MLP, and a key and value head for each query head. Llama changed all three. We test each change by lesson 14's method.

## The idea

RMSNorm replaces layer normalization. Lesson 9's normalization subtracts a vector's mean, divides by its spread, then applies a gain and a shift. RMSNorm has two steps: it divides the vector by the root of the mean of its squares, then applies a gain. It subtracts no mean and adds no shift.

SwiGLU replaces the MLP. Lesson 9's MLP widens, bends and narrows. SwiGLU widens twice, with two separate grids. One result passes through a bend and is multiplied, entry by entry, with the other. The bent result scales each entry of the other, and can shrink it to 0 or flip its sign. Then one grid narrows the product. The bend is called SiLU: x times sigmoid(x), where sigmoid(x) = 1 / (1 + e to the -x) squashes any number into the range 0 to 1.

Three grids cost more than two, so we make each narrower: 8/3 of the width in place of 4 times. Three grids at 8/3 hold as many weights as two at 4: 3 × 8/3 = 2 × 4. At a width of 128 that is 130,944 parameters against the old 131,712, which included 640 biases.

Grouped heads share keys and values. Lesson 7 gave each head its own queries, keys and values. Here several query heads read from one key head and one value head: our 4 query heads read from 2. The grids that make keys and values halve. So does the memory a model needs if it keeps each token's keys and values while it writes text.

The MLP's linear layers also lose their biases, as attention's did in lesson 15.

That is the layout of Llama, a family of open models from Meta. Hugging Face's `transformers` library has an implementation. We used it to build a tiny one with random weights: 2 blocks, a width of 48, 4 heads, 2 key heads and a vocabulary of 64 tokens. We saved its weights and the scores it gives for 40 random tokens. Our model, loaded with those weights, picks the same top token at each of the 40 places, and its 64 scores at the last place match to 0.00001.

Meta's released Llamas would not load as they are. Our code hard-codes three settings: the width of SwiGLU, the small number inside RMSNorm, and the base of the rotary speeds. Llama 2 sets the first two differently and Llama 3.2 all three. Llama 2 also keeps a separate grid for the scores.

## By hand

RMSNorm on the vector (3, 4). The squares are 9 and 16, their mean is 12.5, and its root is 3.536. The result is (0.849, 1.131). Layer normalization gives (-1, 1) for the same vector, and also for (103, 104), because it subtracts the mean. RMSNorm does not, and gives (0.995, 1.005) for (103, 104).

SiLU at three points:

| x | sigmoid(x) | SiLU(x) |
|---|---|---|
| -1 | 0.269 | -0.269 |
| 0 | 0.5 | 0 |
| 1 | 0.731 | 0.731 |

SwiGLU's hidden size at a width of 128: 8 × 128 / 3 = 341, rounded down.

## The code

```python
# gpt/model.py
    def forward(self, x):
        """Divide each vector by the root of the mean of its squares. No mean is subtracted."""
        return self.gain * x / (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
```

```python
# gpt/model.py
    def forward(self, x):
        gate = self.gate(x)
        return self.narrow(gate * torch.sigmoid(gate) * self.wide(x))
```

Grouped heads are two lines in attention. `k` and `v` arrive with `kv_heads` heads, and each is repeated until there is one for each query head:

```python
# gpt/model.py
        share = self.heads // self.kv_heads  # query heads that read each key head
        k, v = k.repeat_interleave(share, dim=1), v.repeat_interleave(share, dim=1)
```

`repeat_interleave(2, dim=1)` turns heads (a, b) into (a, a, b, b).

The model takes its parts as arguments, so an experiment can put one old part back. `kv_heads=None` gives each query head a key head of its own:

```python
# gpt/model.py
class GPT(nn.Module):
    def __init__(
        self, vocab, context, width, heads, layers, kv_heads=None, norm=RMSNorm, mlp=SwiGLU
    ):
```

## The proof

```bash
uv run pytest tests/test_model.py
```

- RMSNorm gives (0.849, 1.131) for (3, 4), equals PyTorch's, and keeps the mean that layer normalization removes.
- SwiGLU equals its formula, and SiLU(1) is 0.731.
- SwiGLU has 130,944 parameters at a width of 128, and the old MLP 131,712.
- Grouped heads equal full heads whose key and value grids hold each head twice.
- The parameter count follows the formula.
- The model starts with no opinion and cannot read the future, and the two layers that add into the stream start smaller than the rest.
- With the reference Llama's weights it gives the reference scores.

```bash
uv run python -m gpt.model
```

```
lesson 15's model, five seeds: 1.8298 1.8259 1.8414 1.8223 1.8217
lesson 15's model: 799,616 parameters
RMSNorm in place of layer normalization: -0.0018 nats, error 0.0025, ratio 0.7, 798,464 parameters
SwiGLU in place of the MLP: -0.0802 nats, error 0.0063, ratio 12.8, 796,544 parameters
2 key heads in place of 4: +0.0037 nats, error 0.0052, ratio 0.7, 734,080 parameters
all three: -0.0763 nats, error 0.0030, ratio 25.6, 729,856 parameters
all three, 5,000 steps, unseen text: 1.531 nats, 2.209 bits per byte
```

Each comparison is lesson 14's method, 600 steps on five seeds. The first three change one part of lesson 15's model, and the last changes all three.

RMSNorm's difference is 0.7 errors from zero. SwiGLU is a real gain at 600 steps: 0.080 nats, 12.8 errors from zero. Grouped heads differ by 0.7 errors and remove 8% of the parameters: 734,080 against 799,616. All three together are ahead by 0.076 nats.

Today's model has 9% fewer parameters than lesson 15's, and at 5,000 steps it scores 2.209 bits per byte on unseen text against lesson 15's 2.210.

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| MLP, eight characters back | 2.663 |
| GPT, 64 characters back | 3.017 |
| GPT, with AdamW | 2.372 |
| GPT, with the loop, 5,000 steps | 2.206 |
| GPT, rotary positions | 2.210 |
| GPT, today's parts | 2.209 |

## What it still cannot do

```
Once upon a time but fortune and 'I,
She is my return waves; and play twice king;
Look the senateward, he found you speak,
To then he di
```

At 5,000 steps, on one seed each, we could not tell the designs apart. Longer runs on more seeds would settle it, and we have not counted what a run costs: the memory it needs and the multiplies a step takes.

## Check yourself

Predict before you run: a model has 8 query heads and 2 key heads. How many query heads read each key head, and by what factor do the key and value grids shrink?

<details><summary>Answer</summary>

Four query heads read each key head, and the two grids are a quarter of their full size.
</details>

<details><summary>What does RMSNorm give for the vector (6, 8)?</summary>

The squares are 36 and 64, their mean is 50, and its root is 7.07. The result is (0.849, 1.131), the same as for (3, 4): the size is gone and the direction is kept.
</details>

<details><summary>In SwiGLU the gate's result is 0 for one entry. What comes out of that entry?</summary>

0 times sigmoid(0) is 0, so that entry of the product is 0 whatever the other grid gives.
</details>

<details><summary>From lesson 14: RMSNorm's difference was under 3 errors from zero. What can we say about it?</summary>

That these runs show no difference. They do not show that there is none: a small one would need more seeds to find.
</details>

Go deeper:

- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (2019). RMSNorm.
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020). Where SwiGLU comes from.
- Ainslie and others, [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (2023). Grouped heads, and what they cost in quality.
