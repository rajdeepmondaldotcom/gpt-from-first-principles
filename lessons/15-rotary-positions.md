# 15. Rotary positions

By the end of this page the model has no table of places, and it runs on text four times longer than it trained on.

Lesson 8's table has one row for each place, 64 of them, so a 65th token cannot be read. A test shows the error. Each place also has a row of its own, so what the model learns about places 3 and 5 says nothing about places 43 and 45.

## The idea

Attention needs to know how far apart two tokens are. Lesson 8 put the place into the token's vector once, before any layer. Rotary positions put it where it is used: into each query and each key, at each attention layer.

Take a query's entries two at a time. A pair of numbers is a point on a flat surface. Turn that point around the centre by an angle of place × speed. Do the same to each key. A turn moves a point and keeps its distance from the centre, so no vector changes length.

A score is a dot product. For one pair of entries, the dot product of two points depends on their lengths and on the angle between them. Turn the query by m × speed and the key by n × speed, and the angle between them changes by (m - n) × speed. The places m and n are gone. Only the distance between them is left. Move both tokens 100 places on and the score is the same.

One speed is not enough, because an angle comes back to where it started after a full turn. At 1 radian a place, a distance of 44 is seven full turns and 0.018 radians, which looks like a distance of 0. So each pair gets a speed of its own. The first turns 1 radian a place and the speeds fall, pair by pair, toward one ten-thousandth of that. Fast pairs tell neighbours apart. Slow pairs tell near from far.

Values are not turned. The turn exists to set the scores.

Nothing here has a size tied to the length of the text. The model can be handed more tokens than it trained on. Whether it reads them well is a separate question, and the run below answers it.

## By hand

One pair, with a speed of 1. To turn a point (x, y) by an angle a:

(x cos a - y sin a, x sin a + y cos a)

A query (1, 0) at place 2 turns by 2 radians to (cos 2, sin 2) = (-0.416, 0.909). A key (1, 0) at place 1 turns by 1 radian to (0.540, 0.841).

Score: -0.416 × 0.540 + 0.909 × 0.841 = 0.540

That is cos 1, the cosine of the distance. Put the same two vectors at places 102 and 101 and the score is cos 1 again.

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

`gpt/model.py` holds this lesson's model: lesson 10's GPT with no table of places and with this attention. `seen` and `kv_heads` belong to later lessons. Here `seen` is 0 and `kv_heads` equals `heads`. The model takes its normalization and its MLP as arguments, and here we pass lesson 9's.

## The proof

```bash
uv run pytest tests/test_rope.py
```

- The hand example: place 0 does not turn, and place 1 turns (1, 0) to (0.540, 0.841).
- The second pair turns at a hundredth of the first pair's speed when there are two pairs.
- Turning keeps each vector's length.
- A score for places 5 and 2 equals the score for places 105 and 102, and for 1,005 and 1,002. Places 5 and 3 give a different score.
- Shuffle the earlier tokens and the last token's result changes, with no table of places.
- A model with a context of 16 runs on 200 tokens. Lesson 10's model raises an error.

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

The first three lines are lesson 14's method, 600 steps on five seeds. The rotary model is ahead by 0.091 nats, 10.2 errors from zero. Two things differ between the models, so the gain cannot be given to rotation alone: the table is gone, and this attention has no biases.

The gain does not last. After lesson 12's 5,000 steps the rotary model scores 2.210 bits per byte, and lesson 12's scored 2.206.

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

The model runs on 256 tokens, and it reads them worse: 1.532 nats at 64 tokens, 1.869 at 128 and 2.393 at 256. It trained on distances up to 63 and never met a longer one. Rotary positions remove the limit on length. They do not teach the model what a longer text looks like.

The rest of the model is still 2019's: layer normalization, a GELU MLP, and a key and value head for each query head.

## Check yourself

Predict before you run: lesson 8's table of places held 64 × 128 = 8,192 parameters. How many does `rope` hold?

<details><summary>Answer</summary>

None. The speeds are fixed numbers, worked out when the function runs.
</details>

<details><summary>Two tokens sit at places 10 and 7. Two others, with the same vectors, sit at places 500 and 497. How do their scores compare?</summary>

They are equal. Both pairs are 3 apart, and only the distance enters the score.
</details>

<details><summary>Why does the slowest pair matter for a text of 1,000 tokens?</summary>

A fast pair has turned through many full circles by then, so it cannot tell a distance of 1,000 from a short one. A slow pair has turned a fraction of a circle and still can.
</details>

<details><summary>From lesson 8: shuffling the earlier tokens left the last token's result alone until places were added. What does the same test show here?</summary>

The result changes. Turning each key by its place gives order, with no table and no parameters.
</details>

Go deeper:

- Su and others, [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021). The paper.
- EleutherAI, [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/). The same idea with pictures of the turning pairs.
- Kazemnejad and others, [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466) (2023). How each way of marking position fares on text longer than it trained on.
