# 6. Attention

By the end of this page each token can choose what to read from the others, and one set of weights serves text of any length.

Our model reads eight tokens through eight fixed slots. The weights that read the third slot share nothing with the weights that read the fifth, so a pattern learned in one place has to be learned again in the next. The window's length is also built into the first layer.

## The idea

We want each token to choose what to read.

Each token is a row of numbers, its embedding. A row of numbers is called a vector, and the count of numbers in it is its width. To compare two vectors, multiply them entry by entry and add the products. This is the dot product. It is large when the two vectors have large numbers of the same sign in the same places.

A first try: for one token, take its dot product with each token in the text, turn those scores into weights with softmax, and replace the token with the weighted average of the embeddings. This lets a token read from anywhere. It has no parameters beyond the embeddings, so it cannot learn what to look for.

Give each token three vectors, made from its embedding by three linear layers:

- a query: what this token is looking for
- a key: what this token offers
- a value: what this token hands over when chosen

The score for token i reading token j is the dot product of i's query with j's key. Softmax turns token i's scores into weights that sum to 1. Token i's result is the weighted sum of the values. This is attention. The weights are called attention weights. They are worked out afresh for each text and are not parameters, unlike the weights inside a layer.

The three linear layers are the same for each token, wherever it stands. The fixed slots are gone, and the same layers work on any number of tokens. Two tests below check both.

Call the usual distance of a set of numbers from their average their spread. Statisticians call it the standard deviation. A dot product adds one product per entry, and the spread of a sum of n random terms grows like the square root of n. A test measures it: with random vectors at a width of 256, the scores have a spread near 16. Among 100 tokens, softmax then puts more than 0.8 of the weight on a single token, on average, so each token reads from one place. So we divide the scores by the square root of the width. The spread drops to about 1 and the largest weight to under 0.1.

## By hand

Three tokens: `the`, `cat`, `sat`. We work out what `sat` reads. Keys have width 4 and values have width 2.

| Token | Key | Value |
|---|---|---|
| `the` | (1, 0, 1, 0) | (1, 0) |
| `cat` | (0, 2, 0, 2) | (0, 1) |
| `sat` | (1, 1, 0, 0) | (1, 1) |

The query for `sat` is (0, 2, 0, 2).

| | `the` | `cat` | `sat` |
|---|---|---|---|
| Query · key | 0 | 8 | 2 |
| Divided by √4 = 2 | 0 | 4 | 1 |
| e to that power | 1 | 54.598 | 2.718 |
| Weight (divide by 58.316) | 0.017 | 0.936 | 0.047 |

The result is 0.017 × (1, 0) + 0.936 × (0, 1) + 0.047 × (1, 1) = (0.064, 0.983). `sat` took 94% of what it read from `cat`.

Now all three tokens at once. Stack the queries as the rows of a grid Q, the keys as K and the values as V. `K.T` is K turned on its side, rows for columns. `Q @ K.T` then holds each query's dot product with each key, a grid of 3 × 3. Row i is what token i reads. The row for `sat` is the row we worked out, and a test finds it there.

## The code

```python
# gpt/attention.py
def attend(q, k, v, mask=None):
    """Each query scores each key, the scores become weights, and the weights mix the values."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        # shortcut: a row with every key masked comes out as NaN; a causal mask never has one.
        scores = scores.masked_fill(~mask, -math.inf)  # a weight of 0 wherever mask is False
    weights = softmax(scores)
    return weights @ v, weights
```

`k.transpose(-2, -1)` turns K on its side, as `.T` does, and also works when there are more than two directions. `q.size(-1)` is the width. `softmax` is ours from lesson 3. Leave `mask` for the next lesson.

## The proof

```bash
uv run pytest tests/test_attention.py
```

- The hand example gives the scores, the weights and the result above.
- The hand example is one row of the whole grid.
- `attend` equals PyTorch's `scaled_dot_product_attention`.
- At a width of 256, with 100 tokens, undivided scores have a spread near 16 and the largest weight averages above 0.8. Divided, the spread is near 1 and the largest weight is under 0.1.
- No slots: shuffle the tokens and the results shuffle with them.
- The same weights serve 3 tokens and 30.

## What it still cannot do

Lesson 4's model made one guess per window. Attention gives a result at each place, so we could ask for a guess at each place: from the result at place 3, guess the token at place 4. As it stands that would be cheating. The token at place 3 can read the token at place 4, which is the answer. A model trained that way would be no use when it writes, because then the next token does not exist yet.

Each token also gets one set of weights, so one view of the text. To continue `the cat sat on the`, a token may need the last noun and the last verb at once.

## Check yourself

Predict before you run: at a width of 64, how wide do the undivided scores spread?

<details><summary>Answer</summary>

About 8, the square root of 64.
</details>

<details><summary>Five tokens, width 16. What shape is the grid of weights, and what does each row sum to?</summary>

5 × 5. Each row sums to 1, because softmax runs along the row.
</details>

<details><summary>Queries and keys must have the same width. Values can have any width. Why?</summary>

A dot product pairs up the entries of a query and a key, so they need the same count. Values are only added together, weighted.
</details>

<details><summary>All the keys are equal. What does each token read?</summary>

Each score in a row is the same, so each weight is equal and the result is the plain average of the values.
</details>

<details><summary>From lesson 3: add 10 to each score in the row (0, 4, 1). What happens to the weights?</summary>

Nothing. Softmax depends only on the gaps between scores.
</details>

Go deeper:

- Vaswani and others, [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017). Section 3.2.1 is this page.
- Bahdanau, Cho and Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014). The paper that brought attention to translation.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Queries, keys and values in pictures.
