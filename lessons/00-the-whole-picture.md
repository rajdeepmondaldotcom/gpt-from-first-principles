# 0. The whole picture

By the end of this page you know where the next 45 lessons go, and what each one is for.

First we follow the finished storyteller from one end to the other.

## The idea

You are going to build a storyteller. Someone types a name and three words. A model that you trained writes a new story with them, the computer reads it aloud, and the listener types what happens next. It runs on a server that you wrote and serves many listeners at once. Then you will hand the same code a pile of chess games in place of stories, and it will learn the moves of chess. Last, you rent a graphics card. The same code trains bigger models there, an open engine called vLLM serves one, and a chess program called Stockfish rates your player.

Each lesson adds a piece or tests one, in this order.

| Lessons | Stage | The question it answers |
|---|---|---|
| 1 | The task | What is a language model, and how do we score one? |
| 2 to 4 | Learning | How does a model get better by itself? |
| 5 | Text | What does it read? |
| 6 to 10 | The transformer | What is inside it? |
| 11 to 14 | Training | How do you train one, and know that it worked? |
| 15 and 16 | Today's model | What has changed since 2019? |
| 17 to 21 | A real model | What will it cost and read, how big should it be, and can we train it? |
| 22 to 24 | Using and measuring | What does it write, how fast, and how good is it? |
| 25 to 30 | Adapting | How do you teach it a job, a taste and a rule? |
| 31 to 37 | Serving | How do many people use it at once, and how fast? |
| 38 to 40 | A different world | Was any of this about English? |
| 41 and 42 | Real hardware | What does a rented machine buy? |
| 43 | Splitting the model | What if no one card holds the model? |
| 44 and 45 | A stronger player | How strong can the player get? |

The model answers `Once upon a time`, and from lesson 25 an order, so you can watch it learn to write. Up to lesson 24, and again in lessons 41 and 42, we score in bits per byte, a unit lesson 1 defines, on Shakespeare and then on stories. Lessons 25 to 40 and 43 to 45 use measures of their own, such as tokens a second and legal moves. Each lesson opens on something the last one left undone.

Each claim on a page is one of three things: a test that passes, a number that a script printed, or a source you can open. If an idea does not pay off on a laptop, we say so, and lessons 41 and 42 try some of them again on a rented card. Each dollar figure is the hours AWS reported times its price list.

The lessons are hard on purpose: you work sums by hand before the code does them, and guess each result before you see it. Each page asks one question that begins "Predict before you run". Where the run answers it, the question stands above the run. Write your guess down before you read on.

## By hand

One request, followed through the finished storyteller. The listener types Mira, cat, moon and brave.

| Step | What happens | Lesson |
|---|---|---|
| 1 | The page sends `Name: Mira`, `Words: cat, moon, brave`, `Story:` to the server | 36 |
| 2 | The text is cut into 20 tokens: `N`, `ame`, `:`, ` M`, `ir`, `a` and so on | 5, 18 |
| 3 | The engine checks its memory. This request, 20 tokens in and up to 120 out, may come to need (20 + 120) × 6,144 = 860,160 bytes | 23, 32 |
| 4 | The model reads the 20 tokens in one pass through 6 blocks. In each, each token chooses what to read from the tokens before it | 6 to 10, 15, 16 |
| 5 | The keys and values of those 20 tokens are kept, so they are never worked out again | 23 |
| 6 | The last place gives one score for each token that could come next | 10 |
| 7 | The scores become chances, and one token is drawn | 2, 22 |
| 8 | That token goes back in alone. Other listeners' tokens ride the same pass | 23, 32 |
| 9 | Tokens become bytes and bytes become text. Half a character waits for its other half | 36 |
| 10 | The browser speaks each sentence as it ends. The end token arrives, and the request leaves the batch | 25, 36 |

You will work out each number in that table. The model's 5,374,464 weights came from a run that read 73 million tokens of stories, then from 2,000 steps on stories written to order, and 200 more on pairs of stories with one judged the better.

## The code

Each lesson has a short file in `gpt/`, a test in `tests/` and a page in `lessons/`. A later file uses the earlier ones, and never changes a line that an earlier page quotes. The whole of attention, from lesson 6, shows the size of the pieces:

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

You need not follow it yet. By lesson 6 you will know each line of it.

## The proof

```bash
uv sync
```

```bash
uv run pytest
```

The first command installs what the code needs. The second runs each lesson's tests on the processor in a few minutes.

A test here is a claim that can fail. For each idea, the lesson names what a wrong line would cause, and the test checks that it does not happen. Break a line the way the lesson warns about and its test goes red.

## What it still cannot do

You have built none of it yet. Lesson 1 starts with a text file and a loop that counts characters.

## Check yourself

Each lesson ends with questions like these. Guess each one in writing, and then look.

Predict before you run: the model has 4,096 tokens for pieces of text. How many scores does the last place give?

<details><summary>Answer</summary>

4,097: one for each of 4,096 tokens, and one for the token that ends a story.
</details>

Then come questions with hidden answers. One of them asks about an earlier lesson.

<details><summary>A request has a prompt of 20 tokens and may write 120 more. Each token kept costs 6,144 bytes. How much memory can the request come to need?</summary>

140 × 6,144 = 860,160 bytes.
</details>

Go deeper:

- [MAP.md](../MAP.md). What the lessons leave out, and where to learn each thing.
- [PRACTICE.md](../PRACTICE.md). Nine routines for the skills that reading cannot give you.
- [GLOSSARY.md](../GLOSSARY.md). Each term, with the lesson that defines it, and our plain words beside the field's.
