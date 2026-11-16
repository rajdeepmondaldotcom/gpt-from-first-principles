# 1. Predict the next token

By the end of this page you have a program that writes text and a number that says how good it is.

A language model does one job. You give it some text, and it says how likely each possible next piece is. The pieces are called tokens. For now a token is one character.

We have a file of Shakespeare: 1,115,394 characters, 65 different ones. We want a model that reads the file and then guesses what comes next in text it has never seen. We also need a score for those guesses, or we cannot tell a good model from a bad one.

## The idea

Count. Go through the file, and for each character write down which character came next. In the text we count on, `u` follows `q` 563 times out of 563. After a space, `t` comes next 21,591 times and `z` 9 times. Divide each row of counts by its total and you have a table of probabilities: for each character, the chance of each next character. A model that looks one character back is called a bigram.

Score the model by its surprise. If the model gave the true next character a probability of p, its surprise is minus the logarithm of p. Probability 1 costs nothing. Probability 1/2 costs ln 2 = 0.693. Probability 0 costs infinity. The model's loss is its average surprise over a text, and lower is better. This loss is called cross-entropy. Each model in this course trains to reduce it.

The logarithm is there for a reason. The chance a model gives to a whole text is the product of the chances it gave to each character. Logarithms turn that product into a sum, so lowering the loss and raising the probability of the text are the same act. The counted table gives the training text a higher probability than any other table can. That makes it the maximum-likelihood estimate, and a test below checks it.

Natural logarithms measure surprise in nats. Divide by ln 2 to get bits. Our file is plain ASCII, one byte per character, so the loss in bits is bits per byte. That number is the scoreboard for the whole course.

We need a yardstick. A model that knows nothing gives each of the 65 characters the same chance. Its surprise is ln 65 = 4.174 nats on any character, which is 6.022 bits. A model that learned something beats 6.022.

We count on the first 90% of the file and score on the last 10%. A model can look good on its own training text by memorising it, so we report the score on the held-out part.

Held-out text sets a trap. Suppose a pair of characters never occurs in the first 90% and does occur in the last 10%. The table gives it probability 0, the surprise is infinite, and so is the average. We add 1 to each count before dividing, which leaves no zero in the table.

One more fact explains what training can and cannot do. Say the true chances of the next character are p and the model says q. The loss splits into two parts:

cross-entropy(p, q) = entropy(p) + KL(p, q)

Entropy is the surprise left over for a perfect model, because text is partly random. KL, the Kullback-Leibler divergence, is the extra cost of q being wrong. It is zero only when q equals p. Training shrinks the second part and can never touch the first.

## By hand

Take the sentence `the cat sat on the mat`. It has 22 characters, so 21 pairs, and 10 different characters.

| After | Comes | Probability |
|---|---|---|
| `a` | `t`, all three times | 1 |
| `t` | `h` twice, space twice | 1/2 each |
| space | `c`, `s`, `o`, `t`, `m` once each | 1/5 each |
| any other character | one thing only | 1 |

Score the table on its own sentence. Twelve pairs had probability 1 and cost nothing. Four had probability 1/2 and cost ln 2 each. Five had probability 1/5 and cost ln 5 each.

(4 × 0.693 + 5 × 1.609) / 21 = 0.515 nats

A blind guess over 10 characters costs ln 10 = 2.303 nats. Counting cut the surprise by more than four times.

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

`pairwise(text)` gives each character with the one after it. To write text, start from a character, draw the next one from its row, and repeat:

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

## The proof

```bash
uv run pytest tests/test_bigram.py
```

The tests check each claim on this page:

- A model that knows nothing scores ln 65.
- Each row of the table sums to 1.
- The hand example gives 1, 1/2 and 0.515 nats.
- Counting gives the most likely table. Take the counted table, move some probability from one next character to another, and the loss on the training text rises. The test tries 200 such moves.
- Cross-entropy equals entropy plus KL.
- An unseen pair costs infinity until we add 1 to the counts.

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

## What it still cannot do

Ask it to continue `Once upon a time`:

```
Once upon a timeron:
Foleithathavout:
Whe irtthogor s.
```

It learned that a name, a colon and a new line open a speech. It cannot spell, because it picks each letter knowing only the one before.

More context needs a bigger table. Two characters back means 65 × 65 rows. Ten back means 65 to the power 10 rows, far more rows than the file has characters, so nearly all of them stay empty.

Counting also has nowhere to go. It gives its answer in one pass, with nothing left to adjust. We need a model that starts out wrong and gets better in small steps, whatever its shape.

## Check yourself

Predict before you run: the bigram scores 2.455 nats on its training text. Will the unseen text score higher or lower, and by a lot or a little?

<details><summary>A model gives the true next character probability 1/4 on each step. What is its loss in nats and in bits?</summary>

ln 4 = 1.386 nats, which is 2 bits.
</details>

<details><summary>We add 1 to each count. What happens to the loss if we add 1,000 instead?</summary>

The added counts swamp the real ones. Each row moves toward equal chances, and the loss climbs toward ln 65.
</details>

<details><summary>A model scores 0 on a text. What do you know?</summary>

It gave probability 1 to each next character. Either the text has no randomness, or the model memorised it. Score it on unseen text to find out which.
</details>

Go deeper:

- Claude Shannon, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (1948). Section 3 builds this same model of English by hand.
- Andrej Karpathy, [The spelled-out intro to language modeling](https://www.youtube.com/watch?v=PaCmpygFfXo). The bigram model, built slowly on names.
- Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), chapter 5. Loss functions as maximum likelihood.
