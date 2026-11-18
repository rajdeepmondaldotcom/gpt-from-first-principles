# 4. A neural language model

By the end of this page the model reads eight characters back and writes things that look like words.

The bigram sees one character back. A table for eight characters back needs 65 to the power 8 rows, more than 300 trillion. The file has a million characters, so nearly all of those rows would stay empty.

## The idea

Replace the table with a function. Eight characters go in, 65 scores come out, and the function has a fixed number of parameters.

Give each character a row of 24 numbers, called its embedding. The rows start random and training moves them. To read eight characters, look up their eight rows and lay them end to end: 192 numbers.

Next comes a linear layer. It turns those 192 numbers into 256 new ones, each a weighted sum of all 192 plus a constant called a bias. The weights sit in a grid `W` of shape 192 × 256 and the biases in a list `b`. The layer is one line, `x @ W + b`. `@` is matrix multiplication: output j is each input times the weight in column j, added up. When `x` holds many rows of inputs, each row is handled the same way.

Two linear layers in a row do no more than one, because a weighted sum of weighted sums is a weighted sum. So between the layers we bend each number with tanh, the function (e to the x, minus e to the -x) divided by (e to the x, plus e to the -x). It squashes any number into the range -1 to 1. The 256 bent numbers are called hidden units.

A second linear layer turns the hidden units into 65 scores, and softmax turns scores into chances. The whole thing is called a multi-layer perceptron, or MLP. Models built from layers like these are called neural networks.

Count the parameters:

| Part | Shape | Parameters |
|---|---|---|
| Embeddings | 65 × 24 | 1,560 |
| First layer | 192 × 256, plus 256 biases | 49,408 |
| Second layer | 256 × 65, plus 65 biases | 16,705 |
| Total | | 67,673 |

The lookup is a matrix multiplication too. Write a character as 65 numbers, all 0 except a 1 at its place, called a one-hot row. Multiply it by the embedding grid and out comes the character's row. So the model is multiply, add and bend, and lesson 2's chain rule handles each. From here on PyTorch's autograd finds the slopes.

The second layer starts at zero. Each score is then 0, each character gets the same chance, and the first loss is ln 65: the model starts with no opinion.

The other parts start as small random numbers. They must differ: two hidden units that start with equal weights get equal slopes and stay equal for ever.

They must be small because of the shape of tanh. Its slope is 1 minus the square of its output: 1 at an output of 0, and 0.02 at an output of 0.99. The chain rule multiplies along the path, so each weight that feeds such a unit gets a fiftieth of the slope. Call a unit stuck when its output is beyond 0.99 either way. PyTorch starts each first-layer weight within ±1 / √192, about ±0.072. Start the first layer 20 times too large and 82% of the hidden units begin stuck.

## By hand

A layer with 2 inputs and 3 outputs. The input is x = (2, 1) and

```
W = 1  0  -1
    0  3   2
```

Each output takes one column of W:

| Output | Sum | Result |
|---|---|---|
| First | 2 × 1 + 1 × 0 | 2 |
| Second | 2 × 0 + 1 × 3 | 3 |
| Third | 2 × -1 + 1 × 2 | 0 |

Now bend. tanh(2) = 0.964 with slope 1 - 0.964² = 0.071. tanh(3) = 0.995 with slope 0.010. tanh(0) = 0 with slope 1.

## The code

```python
# gpt/mlp_lm.py
class MLP(nn.Module):
    def __init__(self, vocab, context=8, width=24, hidden=256):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, width)  # a row of `width` numbers per character
        self.hidden = nn.Linear(context * width, hidden)
        self.out = nn.Linear(hidden, vocab)
        nn.init.zeros_(self.out.weight)  # start with no opinion: every score 0
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        """x holds character ids, shape (batch, context). Returns scores, shape (batch, vocab)."""
        rows = self.embed(x)  # (batch, context, width)
        h = torch.tanh(self.hidden(rows.flatten(1)))  # (batch, hidden)
        return self.out(h)
```

A class that builds on `nn.Module` is a model. PyTorch finds its parameters through `model.parameters()`, and calling `model(x)` runs `forward`. The line `super().__init__()` sets that up.

`nn.Embedding` is the lookup and `nn.Linear` is `x @ W + b`. PyTorch stores `W` turned on its side, as 256 × 192, and does the turning itself. `rows.flatten(1)` lays the eight rows end to end.

A batch is a set of random windows from the text, each with the character that followed it:

```python
# gpt/mlp_lm.py
def batch(ids, context, size, generator):
    """Pick `size` random windows: `context` characters in, the character after them out."""
    at = torch.randint(len(ids) - context, (size,), generator=generator)
    return ids[at[:, None] + torch.arange(context)], ids[at + context]
```

`torch.randint` picks `size` random starting positions. `at[:, None]` stands them up as a column, and adding the row 0, 1, up to 7 broadcasts into a grid of positions, one window per row.

The loop is lesson 3's, with autograd in place of the hand-worked slopes:

```python
# gpt/mlp_lm.py
        x, y = batch(ids, model.context, 64, generator)
        loss = cross_entropy(model(x), y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= rate * p.grad
```

`zero_grad` clears the old slopes, as lesson 2 did by hand. Inside `torch.no_grad()` PyTorch stops recording operations, because the step itself is not part of the loss.

## The proof

```bash
uv run pytest tests/test_mlp_lm.py
```

- The hand example gives (2, 3, 0), and the slope of tanh matches nudging.
- A lookup equals a one-hot multiply, and `nn.Linear` equals multiply then add.
- The first loss is ln 65.
- Hidden units that start equal stay equal.
- A first layer 20 times too large starts with more than half its hidden units stuck, and its slopes come out more than 3 times smaller.
- After 2,000 steps the model beats the bigram on unseen text.

```bash
uv run python -m gpt.mlp_lm
```

```
normal start:    0% of hidden units stuck, 2.302 nats after 2,000 steps
20x bigger start: 82% of hidden units stuck, 2.652 nats
parameters: 67,673
loss before training: 4.174 nats (ln 65 = 4.174)
after 30,000 steps, first 100,000 training characters: 1.657 nats
unseen text:   1.846 nats, 2.663 bits per byte
```

The training loss is measured on the training text's first 100,000 characters.

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |
| Bigram, learned | 3.599 |
| MLP, eight characters back | 2.663 |

## What it still cannot do

```
Once upon a time it plain? till I to susing to your me that was pifis last q
```

`till`, `your` and `last` are words. `susing` and `pifis` are not, and the words do not add up to a sentence.

Eight characters is about a word and a half. The model cannot see further back than that. Start the prompt with `Twice` in place of `Once` and, with the same random draws, it writes the same continuation, because the last eight characters did not change.

Each place in the window also has weights of its own. The 6,144 weights that read the third character share nothing with those that read the fifth, so a pattern learned at one place has to be learned again at every other.

## Check yourself

Predict before you run: widen the window from 8 characters to 16. How many parameters does the first layer have now?

<details><summary>Answer</summary>

16 × 24 × 256 + 256 = 98,560, twice the weights.
</details>

<details><summary>From lesson 1: knowing nothing is an even guess among 65 characters, and scores ln 65. This model scores 1.846 nats. An even guess among how many characters scores that?</summary>

About 6, since e^1.846 = 6.3.
</details>

Go deeper:

- Bengio and others, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (2003). This model, with words as tokens.
- Andrej Karpathy, [Building makemore part 3: activations and gradients](https://www.youtube.com/watch?v=P6sfmUTpUmc). Stuck units and starting scales, measured layer by layer.
