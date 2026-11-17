# 0. The whole picture

By the end of this page you know where the next 42 lessons go, and what each one is for.

Forty-two lessons are easy to get lost in. So before the first one, here is the finished thing, followed from one end to the other.

## The idea

You are going to build a storyteller. Someone types a name and three words. A model that you trained writes a new story with them, the computer reads it aloud, and the listener says what happens next. It runs on a server that you wrote and serves many listeners at once. Then you will hand the same code a pile of chess games in place of stories, and it will learn chess.

Each lesson adds one piece, and the pieces sit along one road.

| Lessons | Stage | The question it answers |
|---|---|---|
| 1 | The task | What is a language model, and how do we score one? |
| 2 to 4 | Learning | How does a model get better by itself? |
| 5 | Text | What does it read? |
| 6 to 10 | The transformer | What is inside it? |
| 11 to 14 | Training | How do you train one, and know that it worked? |
| 15 and 16 | Today's model | What has changed since 2019? |
| 17 to 21 | A real model | What will it cost, what will it read, how big should it be? |
| 22 to 24 | Using and measuring | What does it write, and how good is it? |
| 25 to 30 | Adapting | How do you teach it a job, a taste and a rule? |
| 31 to 37 | Serving | How do many people use it at once? |
| 38 to 40 | A different world | Was any of this about English? |
| 41 and 42 | Real hardware | What does a rented machine buy? |

The road is a loop. What people do with a served model becomes text, and text is where training starts.

Three threads run through the lessons. One prompt, `Once upon a time`, is answered by the model at each stage, so you can watch it learn to write. One scoreboard, in bits per byte, puts each model on the same scale. And each lesson opens on something the last lesson's model could not do.

Each claim on a page is one of three things: a test that passes, a number that a script printed, or a source you can open. Where an idea does not pay off on a laptop, the page says so.

It is hard on purpose. You will work sums by hand before the code does them, and guess each result before you see it.

## By hand

One request, followed through the finished storyteller. The listener types Mira, cat, moon and brave.

| Step | What happens | Lesson |
|---|---|---|
| 1 | The page sends `Name: Mira`, `Words: cat, moon, brave`, `Story:` to the server | 36 |
| 2 | The text is cut into 20 tokens: `N`, `ame`, `:`, ` M`, `ir`, `a` and so on | 5 |
| 3 | The engine checks its memory. This request may come to need (20 + 120) × 6,144 = 860,160 bytes | 17, 32 |
| 4 | The model reads the 20 tokens in one pass through 6 blocks. In each, each token chooses what to read from the tokens before it | 6 to 10, 15, 16 |
| 5 | The keys and values of those 20 tokens are kept, so they are never worked out again | 23 |
| 6 | The last place gives 4,097 scores, one for each token that could come next | 10 |
| 7 | The scores become chances, and one token is drawn | 2, 22 |
| 8 | That token goes back in alone. Other listeners' tokens ride the same pass | 23, 32 |
| 9 | Tokens become bytes and bytes become text. Half a character waits for its other half | 36 |
| 10 | The browser speaks each sentence as it ends. The end token arrives, and the request leaves the batch | 25, 36 |

Each number in that table is one you will be able to work out. The weights that did it, 5,374,464 of them, came from a run that read 73 million tokens of stories, and then from 2,000 steps on stories written to order.

## The code

Each lesson has a short file in `gpt/`, a test in `tests/` and a page in `lessons/`. A later file uses the earlier ones and does not change them. Here is the whole of attention, from lesson 6, to show the size of the pieces:

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

You are not meant to follow it yet. By lesson 6 each word in it will be yours.

## The proof

```bash
uv sync
```

```bash
uv run pytest
```

The first command installs what the code needs. The second runs each lesson's tests on the processor in a few minutes.

A test here is a claim that can fail. For each idea, the lesson says what would be true if the code were wrong, and the test checks that it is not. Break a line the way the lesson warns about and its test goes red. That is the difference between code that runs and code you can trust.

## What it still cannot do

You have built none of it yet. Lesson 1 starts with a text file and a loop that counts letters.

## Check yourself

Each lesson ends like this one. First comes a number to guess before you run the code. Guess it, in writing, and then look. The gap between your guess and the answer is the lesson.

Predict before you run: the model gives a score to each token that could come next. From the table above, how many scores is that?

<details><summary>Answer</summary>

4,097: one for each of 4,096 tokens, and one for the token that ends a story.
</details>

Then come questions with hidden answers. One of them always reaches back to an earlier lesson.

<details><summary>A request has a prompt of 20 tokens and may write 120 more. Each token kept costs 6,144 bytes. How much memory can the request come to need?</summary>

140 × 6,144 = 860,160 bytes.
</details>

Go deeper:

- [MAP.md](../MAP.md). What the lessons leave out, and the best open place to learn each thing.
- [PRACTICE.md](../PRACTICE.md). Eight routines for the skills that reading cannot give you.
- [GLOSSARY.md](../GLOSSARY.md). Each term, with the lesson that defines it, and our plain words beside the field's.
