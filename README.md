# GPT from first principles

Build a GPT yourself. Train it, serve it, then teach the same code chess.

You start with a loop that counts letters. Each lesson adds one idea, in code short enough to read in a sitting, with a test that checks it. You need Python and school maths. It is hard on purpose.

Under construction: lessons 1 to 10 of 42 are here.

## The lessons so far

| # | Lesson | What you have at the end |
|---|---|---|
| 1 | [Predict the next token](lessons/01-predict-the-next-token.md) | A program that writes text, and a number that says how good it is |
| 2 | [Gradients](lessons/02-gradients.md) | A model that teaches itself the table lesson 1 had to count |
| 3 | [Tensors](lessons/03-tensors.md) | The same model learning all of Shakespeare in seconds |
| 4 | [A neural language model](lessons/04-a-neural-language-model.md) | A model that reads eight characters back and writes things that look like words |
| 5 | [Tokens](lessons/05-tokens.md) | A tokenizer checked against OpenAI's, and a window that reaches more than twice as far |
| 6 | [Attention](lessons/06-attention.md) | Tokens that choose what to read, with one set of weights for any length |
| 7 | [Causal multi-head self-attention](lessons/07-causal-multi-head-self-attention.md) | A layer that trains on each place at once without seeing the answers |
| 8 | [Position](lessons/08-position.md) | A model that knows where each token stands |
| 9 | [The block](lessons/09-the-block.md) | A unit that still passes gradients back when stacked 24 deep |
| 10 | [The GPT](lessons/10-the-gpt.md) | One model that runs OpenAI's GPT-2 weights and matches a reference implementation's scores |

## The scoreboard

Each model is scored on the same unseen tenth of Shakespeare, in bits per byte. Lower is better.

| Model | Lesson | Bits per byte |
|---|---|---|
| Knows nothing | 1 | 6.022 |
| Bigram, counted | 1 | 3.581 |
| Bigram, learned | 3 | 3.599 |
| MLP, eight characters back | 4 | 2.663 |
| MLP, eight tokens back, 1,024 tokens | 5 | 2.572 |
| GPT, 64 characters back | 10 | 3.017 |

The GPT scored three times as many targets and still trails the MLP. We trained both with plain gradient descent.

## The same prompt at each stage

```
lesson 1    Once upon a timeron: Foleithathavout: Whe irtthogor s.
lesson 4    Once upon a time it plain? till I to susing to your me that was pifis last q
lesson 10   Once upon a timer deat tilses te grin and frit proner bese aff felling Eved.
```

## Run it

```bash
uv sync
uv run pytest
```

Each lesson has a page in `lessons/`, a short file in `gpt/` and a test in `tests/`. A lesson's numbers come from its own script, for example `uv run python -m gpt.bigram`, and are saved in `results/`.

Two tests compare our code with outside references, and `uv run pytest` leaves them out. They need the `reference` extra, and the second needs GPT-2's weights (see lesson 10):

```bash
uv run --extra reference pytest -m reference
```

## How to practise

Open a file in `gpt/`, delete the body of one function, write it again from memory, and run its test.

## License

Apache-2.0. `data/tiny_shakespeare.txt` is public-domain text.
