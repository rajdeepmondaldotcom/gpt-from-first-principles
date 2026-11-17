# 34. Quantization

By the end of this page each weight inside the blocks takes 8 bits, or 4, and you have measured what each choice costs in memory, in loss and in speed.

To write one token the model reads each of its weights, and each weight is a 32-bit number: 21.5 MB for each token written.

## The idea

A weight does not need 32 bits. Store it as a small whole number, and keep one 32-bit number for its whole row, called the scale, that says how large a step of 1 is.

With 8 bits a whole number runs from -127 to 127. Find the largest weight in the row by size and let it be 127, so the scale is that weight divided by 127. Each weight becomes its value divided by the scale, rounded to the nearest whole number. To use a weight, multiply its number by the scale. Rounding moves a number by at most a half, so no weight moves by more than half a step.

With 4 bits the numbers run from -7 to 7. The steps are 127 / 7 = 18 times as coarse.

Each row has a scale of its own. With one scale for a whole grid, a single large weight anywhere would make the steps coarse for each row.

How far can the error carry? A layer's result is a sum of weights times inputs. Each weight is off by at most half a step. So a row's result is off by at most half a step times the sum of the sizes of the inputs. That is a bound for the worst case, where each error pushes the same way. What the model loses in practice we measure: the loss on 500 unseen stories, before and after, with lesson 24's interval on the difference.

An 8-bit number is a byte, a quarter of 32 bits. Two 4-bit numbers share a byte, an eighth. The scales add one 32-bit number a row.

Speed is a separate question. We store the weights small and turn them back into 32-bit numbers each time a layer runs. That is extra work. Engines built for speed multiply the small numbers as they are, with code written for one kind of chip. Ours does not, so this page claims the memory and reports the speed it measured.

## By hand

One row at 4 bits: 0.6, -1.0, 0.26 and 0.0. The largest size is 1.0, so the scale is 1 / 7 = 0.143.

| Weight | Divided by the scale | Rounded | Times the scale | Moved by |
|---|---|---|---|---|
| 0.6 | 4.2 | 4 | 0.571 | 0.029 |
| -1.0 | -7 | -7 | -1.0 | 0 |
| 0.26 | 1.82 | 2 | 0.286 | 0.026 |
| 0.0 | 0 | 0 | 0.0 | 0 |

Half a step is 0.071, and no weight moved further.

Memory, for a grid of 256 rows of 256 weights.

| Stored as | Bytes for the numbers | Bytes for the scales |
|---|---|---|
| 32 bits | 262,144 | 0 |
| 8 bits | 65,536 | 1,024 |
| 4 bits | 32,768 | 1,024 |

## The code

```python
# gpt/quantize.py
def quantize(grid, bits):
    """Each row becomes small whole numbers and one scale. Returns (numbers, scales)."""
    top = 2 ** (bits - 1) - 1  # 127 for 8 bits, 7 for 4
    scale = (
        grid.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / top
    )  # the row's largest maps to top
    return (grid / scale).round().to(torch.int8), scale


def pack(numbers):
    """Two 4-bit numbers in each byte. Halves the bytes of numbers between -8 and 7."""
    low, high = (numbers.flatten() + 8).to(torch.uint8).view(-1, 2).unbind(dim=1)
    return low | (high << 4)
```

`amax(dim=1)` is the largest value along each row. `clamp(min=1e-12)` keeps a row of zeros from dividing by zero. In `pack`, adding 8 makes each number 0 to 15, which is four bits. `high << 4` moves one number four bits up, and `|` lays the other in the four bits below it.

```python
# gpt/quantize.py
    def forward(self, x):
        numbers = unpack(self.numbers, self.shape) if self.four else self.numbers
        return x @ (numbers * self.scale).T
```

This is the layer that stands in for a linear layer. `shrink` puts one in place of each linear layer inside the blocks. The grid of token rows stays at 32 bits: it is 4.2 MB of the 21.5.

## The proof

```bash
uv run pytest tests/test_quantize.py
```

- The hand example gives 4, -7, 2 and 0, and a scale of 1 / 7.
- At 8 bits and at 4, no weight of a random grid moves by more than half a step.
- A row of zeros stays zero.
- Sixty 4-bit numbers pack into 30 bytes and come back as they were.
- A layer's result moves by no more than the bound, at 8 bits and at 4.
- The numbers of a 256 by 256 grid take a quarter of the bytes at 8 bits and an eighth at 4. The scales add 1,024 bytes.
- A shrunk model's scores move less at 8 bits than at 4, and by under 0.005 at 8.

```bash
uv run python -m gpt.quantize
```

```
32 bits: 21.5 MB, 1.480 nats, 155 tokens a second
8 bits: 8.6 MB, +0.0001 nats (95% interval +0.0000 to +0.0002), 107 tokens a second
8 bits: no weight moved by more than 0.50 of a step
4 bits: 6.4 MB, +0.0303 nats (95% interval +0.0286 to +0.0318), 60 tokens a second
4 bits: no weight moved by more than 0.50 of a step
```

Timings are from an M4 laptop that was running other jobs, and they move from run to run.

At 8 bits the model takes 8.6 MB where it took 21.5, and its loss on 500 unseen stories rises by 0.0001 nats, with an interval from +0.0000 to +0.0002. At 4 bits it takes 6.4 MB and loses 0.0303 nats, and that interval, +0.0286 to +0.0318, is far from zero. For scale, lesson 25's fine-tune cost plain stories 0.069.

Both shrunk models write slower: 107 and 60 tokens a second, against 155. Turning each grid back into 32-bit numbers on each call costs more time than the smaller memory saves. What this page delivers is a smaller model. Speed takes arithmetic on the small numbers themselves.

## What it still cannot do

The model is smaller. It still runs once, top to bottom, for each token it writes.

## Check yourself

Predict before you run: the blocks' grids are 17.3 MB of the model's 21.5 MB. About how large is the whole model with those grids at 8 bits?

<details><summary>Answer</summary>

17.3 / 4 = 4.3 MB for the grids, plus the 4.2 MB that stays at 32 bits: about 8.6 MB with the scales. The whole model shrinks 2.5 times, not 4.
</details>

<details><summary>A row's largest weight is 0.5. How far can one of its weights move at 8 bits, and at 4?</summary>

Half a step. At 8 bits a step is 0.5 / 127 = 0.0039, so 0.002. At 4 bits a step is 0.5 / 7 = 0.071, so 0.036.
</details>

<details><summary>Why does one large weight in a row hurt the other weights in that row?</summary>

It sets the scale. The step grows with it, and each smaller weight is rounded to a coarser grid.
</details>

<details><summary>From lesson 24: the loss rose by a few hundredths of a nat at 4 bits. What makes that more than luck?</summary>

Both models scored the same 500 stories, and the difference was taken token by token. Its 95% interval, drawn story by story, does not hold 0.
</details>

Go deeper:

- Jacob and others, [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) (2017). Arithmetic done on the small numbers themselves.
- Frantar and others, [GPTQ](https://arxiv.org/abs/2210.17323) (2022). Rounding each weight so as to undo the error of the ones rounded before it.
- Han, [MIT 6.5940: TinyML and Efficient AI Computing](https://hanlab.mit.edu/courses/2026-fall-65940). Three of its lectures are on quantization and go much further than this page.
