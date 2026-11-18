# 8. Position

By the end of this page each token's vector says where the token stands.

In one attention layer, the last token reads the earlier tokens as a bag, with nothing to say which came first.

## The idea

Tell each token its place before attention runs. A token's place is also called its position.

A first try is to attach the place as one more number: 0 for the first token, 1 for the second. That gives the model a single number to work with, and its size has no limit: place 500 is 500 times place 1. A layer multiplies it by fixed weights, so later places push ever harder, and a place beyond training is a size the layer never met.

The fix repeats what worked for tokens. A token's id picks a row from a table of embeddings. Let its place pick a row from a second table, with one row per place. Add the two rows:

vector for a token = row for what it is + row for where it stands

Both tables start random and both are learned. Training moves the place rows the same way it moves the token rows.

After the addition, the same token in two places gives two different vectors. The layer makes queries and keys from those vectors, so what a token reads can now depend on where things stand.

The second table has a fixed number of rows. We call that number the context: the most tokens the model can read at once, the length of lesson 4's window. A model with a context of 64 has no row for place 64, so it cannot read a 65th token.

## By hand

Width 2, to keep the sums short.

| | Row |
|---|---|
| Token `dog` | (1, 0) |
| Token `man` | (0, 1) |
| Place 0 | (0.5, 0.5) |
| Place 2 | (-0.5, 0.5) |

`dog` first and `man` third, as in `dog bites man`:

- `dog` at place 0: (1, 0) + (0.5, 0.5) = (1.5, 0.5)
- `man` at place 2: (0, 1) + (-0.5, 0.5) = (-0.5, 1.5)

Swap them, as in `man bites dog`:

- `man` at place 0: (0, 1) + (0.5, 0.5) = (0.5, 1.5)
- `dog` at place 2: (1, 0) + (-0.5, 0.5) = (0.5, 0.5)

Four different vectors. Without the place rows there are two, (1, 0) and (0, 1), in both sentences.

## The code

```python
# gpt/gpt2.py
class Embedding(nn.Module):
    """Lesson 8. A vector for what the token is, plus a vector for where it stands."""

    def __init__(self, vocab, context, width):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)  # one row per place, up to `context` places

    def forward(self, idx):
        places = torch.arange(idx.size(-1), device=idx.device)
        return self.token(idx) + self.position(places)
```

`idx` holds token ids with shape (batch, tokens), and `idx.size(-1)` is the number of tokens. `places` is 0, 1, 2 and so on, one per token. `self.position(places)` has shape (tokens, width), and broadcasting adds it to each text in the batch.

## The proof

```bash
uv run pytest tests/test_gpt2.py
```

- One layer cannot tell order until we add positions. Feed an attention layer the tokens 3, 1, 4, 1, 5, then the same tokens with the first four shuffled. With token rows alone, the result at the last place is the same. With place rows added, it differs.
- The hand example gives the four vectors above.
- The same token gets a different vector in each place.
- Text longer than the table fails: a context of 8 takes 8 tokens and raises an error on 9.

## What it still cannot do

We have the parts of one layer: embeddings with positions, and attention. One round of reading lets a token gather facts from other tokens. To use a fact that another token gathered, it needs a second round, so we want to stack layers.

Attention also only mixes. A token's result is a weighted average of values, passed through one linear layer, with no bend after it. And a plain stack of many such layers passes almost no slope back to the first.

## Check yourself

Predict before you run: a model has a context of 1,024 and a width of 768. How many parameters does the table of places hold?

<details><summary>Answer</summary>

1,024 × 768 = 786,432.
</details>

<details><summary>We add the place row to the token row. What would laying them end to end, as lesson 4 did with its eight characters, cost?</summary>

It would double the width of each vector, and of each layer that reads it. Adding keeps the width.
</details>

<details><summary>A model with a context of 64 is given 65 tokens. What happens, and where?</summary>

`self.position(places)` asks for row 64 of a table whose rows run from 0 to 63. PyTorch raises an IndexError.
</details>

<details><summary>From lesson 7: the mask already stops a token reading later tokens. Why does that not give the model order?</summary>

The mask says which tokens a token may read. Among those it may read, nothing says which came first.
</details>

Go deeper:

- Vaswani and others, [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.5. Position rows made from sine waves, with no training.
- Radford and others, [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018). The first GPT, with the learned table used here.
- Lilian Weng, [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/). The other ways to encode position.
