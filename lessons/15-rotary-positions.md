# 15. Rotary positions

By the end of this page the model has no table of places, and it runs on text four times longer than it trained on.

Lesson 8's table has one row for each place, 64 of them, so the model cannot read a 65th token. Each place also has a row of its own, so training on places 3 and 5 does not change the rows for places 43 and 45.

## The idea

Attention needs to know how far apart two tokens are. Lesson 8 put the place into the token's vector once, before any layer. Rotary positions put it where it is used: into each query and each key, at each attention layer.

Pair up a query's entries: entry i with the entry half the vector's length further on. A pair of numbers is a point on a flat surface. Turn that point around the centre by an angle of place × speed, with the angle in radians: a full turn is 2π, about 6.283. Do the same to each key. We leave the values as they are: place matters only where two tokens meet, in the score, and attention only averages the values. A turn moves a point and keeps its distance from the centre, so no vector changes length.

A score is a dot product. For one pair of entries, the dot product of two points is their two lengths times the cosine of the angle between them, a fact of school geometry that the hand example below shows. Turn the query by m × speed and the key by n × speed, and the angle between them changes by (m - n) × speed. So the places enter the score only through m - n. Move both tokens 100 places on and the score is the same.

One speed is not enough, because an angle comes back to where it started after a full turn. At 1 radian a place, a distance of 44 is seven full turns and 0.018 radians, which looks like a distance of 0. So each pair gets a speed of its own. The first turns 1 radian a place and the speeds fall, pair by pair, toward one ten-thousandth of that: each pair turns at the same fraction of the speed of the pair before it. Fast pairs change a score between neighbours. Slow pairs change it only across long distances.

Nothing here has a size tied to the length of the text. We can hand the model more tokens than it trained on. The run below measures how well it reads them.

## By hand

One pair, with a speed of 1. The cosine and sine of an angle are the two coordinates of the point (1, 0) after a turn by that angle. To turn any point (x, y) by an angle a:

(x cos a - y sin a, x sin a + y cos a)

A query (1, 0) at place 2 turns by 2 radians to (cos 2, sin 2) = (-0.416, 0.909). A key (1, 0) at place 1 turns by 1 radian to (0.540, 0.841).

Score: -0.416 × 0.540 + 0.909 × 0.841 = 0.540

That is cos 1, the cosine of the distance. Put the same two vectors at places 102 and 101 and the score is cos 1 again.

The speeds. With 2 pairs, half is 2, and the speeds are 10000⁰ = 1 and 10000^(-1/2) = 0.01.

## The code

```python
# gpt/rope.py
def rope(x, pos, base=10000):
    """Pair up the entries of x and turn each pair. `pos` holds the place of each token."""
    half = x.size(-1) // 2
    speed = base ** (-torch.arange(half, device=x.device) / half)  # the first pair turns fastest
    angle = pos[:, None] * speed  # (tokens, half): place times speed
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., :half], x[..., half:]  # entry i pairs with entry i + half
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
```

`x[..., :half]` takes the first half of the last direction, whatever comes before it. `torch.cat` joins the two turned halves back into one vector.

Attention calls it on each query and key, and on nothing else:

```python
# gpt/model.py
        pos = torch.arange(seen, seen + T, device=x.device)
        q = rope(self.q(x).view(B, T, self.heads, -1).transpose(1, 2), pos)
        k = rope(self.k(x).view(B, T, self.kv_heads, -1).transpose(1, 2), pos)
```

`gpt/model.py` holds this lesson's model: lesson 10's GPT with no table of places and with this attention, whose layers have no biases. It makes queries and keys with two separate layers, `self.q` and `self.k`, and `-1` in `view` means "work this size out". After `transpose(1, 2)` the queries have the shape (batch, heads, tokens, head width), and `rope` turns pairs in the last direction. `seen` counts tokens read on an earlier call and is 0 here. `kv_heads` is lesson 16's, and here it equals `heads`. The model takes its normalization and its MLP as arguments, and here we pass lesson 9's.

## The proof

```bash
uv run pytest tests/test_rope.py
```

- The hand example: place 0 does not turn, and place 1 turns (1, 0) to (0.540, 0.841).
- The second pair turns at a hundredth of the first pair's speed when there are two pairs.
- Turning keeps each vector's length.
- A score for places 5 and 2 equals the score for places 105 and 102, and for 1,005 and 1,002. Places 5 and 3 give a different score.
- Shuffle the earlier tokens and the last token's result changes, with no table of places.
- A model built for windows of 16 tokens runs on 200. Lesson 10's model raises an error.

```bash
uv run python -m gpt.rope
```

```
table of places, five seeds: 1.8962 1.9195 1.9256 1.9106 1.9433
rotary, five seeds:          1.8298 1.8259 1.8414 1.8223 1.8217
rotary minus table: -0.0908 nats, error 0.0089, ratio 10.2
reading 64 tokens: 1.532 nats, 2.210 bits per byte
reading 128 tokens: 1.869 nats, 2.696 bits per byte
reading 256 tokens: 2.393 nats, 3.452 bits per byte
```

The first three lines are lesson 14's method, 600 steps on five seeds. The rotary model is ahead by 0.091 nats, 10.2 errors from zero. Two things differ between the models, so we cannot credit the gain to rotation alone: the table is gone, and this attention has no biases.

The last three lines are one model trained for lesson 12's 5,000 steps. Read at 64 tokens it scores 2.210 bits per byte, and lesson 12's scored 2.206. That is one run of each, and lesson 14 saw two runs of one setting differ by more. We cannot say the gain lasted, or that it did not.

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| MLP, eight characters back | 2.663 |
| GPT, 64 characters back | 3.017 |
| GPT, with AdamW | 2.372 |
| GPT, with the loop, 5,000 steps | 2.206 |
| GPT, rotary positions | 2.210 |

## What it still cannot do

```
Once upon a time,
It did infect by as he an empressing all!
My good my wife knee; most kill my our own.
And off your worshood of him at
```

The model runs on 256 tokens, and it reads them worse: 1.532 nats at 64 tokens, 1.869 at 128 and 2.393 at 256. It trained on distances up to 63 and never met a longer one. Long-context models change the speeds, by dividing each place by a factor or by raising `base`, so that long distances turn by angles the model met in training. [MAP.md](../MAP.md) has where to read.

The rest of the model is still 2019's: layer normalization, a GELU MLP, and a key and value head for each query head.

## Check yourself

Predict before you run: lesson 8's table of places held 64 × 128 = 8,192 parameters. How many does `rope` hold?

<details><summary>Answer</summary>

None. The speeds are fixed numbers, worked out when the function runs.
</details>

<details><summary>Two tokens sit at places 10 and 7. Two others, with the same vectors, sit at places 500 and 497. How do their scores compare?</summary>

They are equal. Both pairs are 3 apart, and only the distance enters the score.
</details>

<details><summary>From lesson 8: shuffling the earlier tokens left the last token's result alone until places were added. What does the same test show here?</summary>

The result changes. Turning each query and key by its place gives order, with no table and no parameters.
</details>

Go deeper:

- Su and others, [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021). The paper.
- EleutherAI, [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/). The same idea with pictures of the turning pairs.
- Kazemnejad and others, [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466) (2023). How each way of marking position fares on text longer than it trained on.
