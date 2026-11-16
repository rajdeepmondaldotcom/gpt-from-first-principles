# GPT from first principles

Build a GPT yourself. Train it, serve it, then teach the same code chess.

You start with a loop that counts letters. Each lesson adds one idea, in code short enough to read in a sitting, with a test that proves it. You need Python and school maths. It is hard on purpose.

Under construction: lessons 1 to 10 of 42 are here.

## The lessons so far

| # | Lesson | What you have at the end |
|---|---|---|
| 1 | [Predict the next token](lessons/01-predict-the-next-token.md) | A program that writes text, and a number that says how good it is |
| 2 | [Gradients](lessons/02-gradients.md) | A model that teaches itself the table lesson 1 had to count |
| 3 | [Tensors](lessons/03-tensors.md) | The same model learning all of Shakespeare in two seconds |
| 4 | [A neural language model](lessons/04-a-neural-language-model.md) | A model that reads eight characters back and writes things that look like words |
| 5 | [Tokens](lessons/05-tokens.md) | A tokenizer that matches OpenAI's, and a window that reaches twice as far |
| 6 | [Attention](lessons/06-attention.md) | Tokens that choose what to read, with one set of weights for any length |
| 7 | [Causal multi-head self-attention](lessons/07-causal-multi-head-self-attention.md) | A layer that trains on each place at once without seeing the answers |
| 8 | [Position](lessons/08-position.md) | A model that knows where each token stands |
| 9 | [The block](lessons/09-the-block.md) | A unit you can stack 24 deep and still train |
| 10 | [The GPT](lessons/10-the-gpt.md) | One model that runs OpenAI's GPT-2 weights and gives GPT-2's answers |

## The scoreboard

Each model is scored on the same unseen tenth of Shakespeare, in bits per byte. Lower is better.

| Model | Lesson | Bits per byte |
|---|---|---|
| Knows nothing | 1 | 6.022 |
| Bigram, counted | 1 | 3.581 |
| Bigram, learned | 3 | 3.599 |
| MLP, eight characters back, 30,000 steps | 4 | 2.663 |
| MLP, eight tokens back, 30,000 steps | 5 | 2.572 |
| GPT, 64 characters back, 3,000 steps of plain gradient descent | 10 | 3.039 |

The GPT is behind the MLP here. It has had a tenth of the steps, and lesson 10 shows the steps are the weak part.

## The same prompt at each stage

```
lesson 1    Once upon a timeron: Foleithathavout: Whe irtthogor s.
lesson 4    Once upon a time it plain? till I to susing to your me that was pifis last q
lesson 10   Once upon a timerres matilseso forts hads ffitll; Heren se whallll ime pabennty
```

## Run it

```bash
uv sync
uv run pytest
```

Each lesson has a page in `lessons/`, a short file in `gpt/` and a test in `tests/`. A lesson's numbers come from its own script, for example `uv run python -m gpt.bigram`, and are saved in `results/`.

Two tests compare our code with outside references and are skipped unless you ask for them. They need `uv sync --extra reference`, and the second needs GPT-2's weights (see lesson 10):

```bash
uv run --extra reference pytest -m reference
```

## How to practise

Open a file in `gpt/`, delete the body of one function, write it again from memory, and run its test.

## License

Apache-2.0. `data/tiny_shakespeare.txt` is public-domain text.
