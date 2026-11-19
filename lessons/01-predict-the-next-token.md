# 1. Predict the next token

By the end of this page you have a program that writes text and a number that says how good it is.

A language model does one job. You give it some text, and it says how likely each possible next piece is. The pieces are called tokens, and the set of different tokens is the vocabulary. For now a token is one character.

We have a file of Shakespeare: 1,115,394 characters, 65 different ones. We want a model that reads the file and then guesses what comes next in text it has never seen, and a way to measure those guesses.

## The idea

Count. Go through the file, and for each character write down which character came next. In the text we count on, `u` follows `q` 563 times out of 563. After a space, `t` comes next 21,591 times and `z` 9 times. Divide each row of counts by its total and you have a table of probabilities: for each character, the chance of each next character.

A model that looks one character back is called a bigram. Fitting a model to text is called training.

To measure the model we need one tool, the logarithm. The natural logarithm of x, written ln x, is the power that a fixed number e, about 2.718, must be raised to in order to give x. So ln 1 = 0 and ln 2 = 0.693. Two rules follow from that:

- ln(a × b) = ln a + ln b
- ln(1/n) = -ln n

Now measure the model by its surprise. If it gave the true next character a probability of p, its surprise is -ln p. Probability 1 costs nothing. Probability 1/2 costs ln 2 = 0.693. Probability 0 costs infinity. The model's loss is its average surprise over a text, and lower is better. This loss is called cross-entropy. Each model in this course trains to reduce it.

The model picks each character from the one before, so the chance it gives to a whole text is those chances multiplied together. The first rule turns that product into a sum. Lowering the loss and raising the probability of the text are then the same act.

The counted table gives the training text a higher probability than any other table can, which makes it the maximum-likelihood estimate. Prince's chapter 5, linked below, has the argument. Its loss on its own text, 0.515 nats for the sentence below, measures how random the next character is once you know the one before: the text's conditional entropy. Any other table loses more, by the KL divergence of its rows from the counted ones. For each row that is the sum, over next characters, of the counted chance times ln(counted chance / its chance), and each row counts as often as its character comes first.

Surprise measured with ln comes in units called nats. With logarithms to base 2 the unit is the bit, and bits = nats / ln 2, because log₂ x = ln x / ln 2. Each character in our file takes one byte of storage, so the loss in bits is also bits per byte. We score every model on that number.

A model that knows nothing gives each of the 65 characters the same chance, 1/65. By the second rule its surprise is ln 65 = 4.174 nats on any character, which is 6.022 bits. A useful model beats 6.022 on unseen text.

We count on the first 90% of the file, the training text, and measure on the last 10%, which the model never saw. A model can look good on its training text by memorising it, so we report the loss on the unseen part.

Suppose a pair of characters never occurs in the first 90% and does occur in the last 10%. The table gives it probability 0, the surprise is infinite, and so is the average. So before dividing we add 1 to each count, which leaves no zero in the table. It is no longer exactly the maximum-likelihood table.

## By hand

Take the sentence `the cat sat on the mat`. It has 22 characters, so 21 pairs, and 10 different characters.

| After | Comes | Probability |
|---|---|---|
| `a` | `t`, all three times | 1 |
| `t` | `h` twice, space twice | 1/2 each |
| space | `c`, `s`, `o`, `t`, `m` once each | 1/5 each |
| any other character | one thing only | 1 |

Measure the table on its own sentence. Twelve pairs cost nothing, four cost ln 2 each and five cost ln 5 each.

(4 × 0.693 + 5 × 1.609) / 21 = 0.515 nats

A blind guess over 10 characters costs ln 10 = 2.303 nats.

## The code

```python
# gpt/bigram.py
def count(text):
    """How often each character follows each other character."""
    return Counter(pairwise(text))


def table(counts, vocab, smooth=0):
    """Turn counts into probabilities. P[a][b] is the chance that b follows a."""
    P = {}
    for a in vocab:
        row = {b: counts[a, b] + smooth for b in vocab}
        total = sum(row.values())
        P[a] = {b: c / total if total else 1 / len(vocab) for b, c in row.items()}
    return P


def loss(P, text):
    """Average surprise in nats: minus the log of the chance the model gave to what came next."""
    pairs = list(pairwise(text))
    return -sum(math.log(P[a][b]) if P[a][b] else -math.inf for a, b in pairs) / len(pairs)
```

`pairwise(text)` gives each character with the one after it. `Counter` is a dictionary that counts, so `counts[a, b]` is how often `b` followed `a`. `math.log` is ln. A row with no counts at all gets equal chances, so it still sums to 1.

To write text, start from a character, pick the next one at random with the chances in its row, and repeat. `rng.choices` does the picking:

```python
# gpt/bigram.py
def sample(P, start, n, rng):
    """Write n more characters, each drawn from the row of the one before it."""
    out = start
    for _ in range(n):
        row = P[out[-1]]
        out += rng.choices(list(row), weights=row.values())[0]
    return out
```

`choices` returns a list of one pick, and `[0]` takes it.

## The proof

```bash
uv run pytest tests/test_bigram.py
```

- A model that knows nothing has a loss of ln 65.
- Each row of the table sums to 1.
- The hand example gives 1, 1/2 and 0.515 nats.
- Move some probability in the counted table from one next character to another, and the loss on the training text rises. The test tries 200 such moves.
- An unseen pair costs infinity until we add 1 to the counts.
- Another table's loss on the sentence is the counted table's plus the KL divergence.

Then run the model on Shakespeare:

```bash
uv run python -m gpt.bigram
```

```
characters: 65
uniform guess: 4.174 nats, 6.022 bits per byte
bigram, training text: 2.455 nats
bigram, unseen text:   2.482 nats, 3.581 bits per byte
```

| Model | Bits per byte on unseen Shakespeare |
|---|---|
| Knows nothing | 6.022 |
| Bigram, counted | 3.581 |

On unseen text the counted bigram scores 2.482 nats, 3.581 bits per byte.

## What it still cannot do

Ask it to continue `Once upon a time`:

```
Once upon a timeron:
Foleithathavout:
Whe irtthogor s.
```

It cannot spell, because it picks each letter knowing only the one before.

More context needs a bigger table. Ten back means 65 to the power 10 rows, far more rows than the file has characters.

Counting also works only for a table. A model of any other shape has nothing to count.

## Check yourself

Predict before you run: `gpt/bigram.py` builds its table with `table(count(train), vocab, smooth=1)`. Change that 1 to 1000. Does the loss on unseen text go up or down?

<details><summary>Answer</summary>

Up, from 2.482 to 3.141 nats. The added counts swamp the real ones, and each row moves toward equal chances.
</details>

<details><summary>A model gives the true next character probability 1/4 on each step. What is its loss in nats and in bits?</summary>

ln 4 = 1.386 nats. Divide by ln 2 = 0.693 to get 2 bits.
</details>

<details><summary>A model has a loss of 0 on a text. What do you know?</summary>

It gave probability 1 to each next character. Either the text has no randomness, or the model memorised it. Measure it on unseen text to find out which.
</details>

Go deeper:

- Claude Shannon, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (1948). Section 3 builds this same model of English by hand.
- Andrej Karpathy, [The spelled-out intro to language modeling](https://www.youtube.com/watch?v=PaCmpygFfXo). The bigram model, built slowly on names.
- Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), chapter 5. Loss functions as maximum likelihood.
