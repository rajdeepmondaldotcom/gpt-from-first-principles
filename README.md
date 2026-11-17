# GPT from first principles

Build a GPT yourself. Train it, serve it, then teach the same code chess.

You start with a loop that counts letters. Each lesson adds one idea, in code short enough to read in a sitting, with a test that checks it. You need Python and school maths. It is hard on purpose.

Under construction: lessons 1 to 24 of 42 are here.

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
| 11 | [Optimizers](lessons/11-optimizers.md) | The same GPT ahead of each earlier model, with only the rule for the step changed |
| 12 | [The training loop](lessons/12-the-training-loop.md) | A run you can stop and resume, bit for bit |
| 13 | [Is it working?](lessons/13-is-it-working.md) | Six checks that catch a silent bug, and a measured cure for a model that stores its text |
| 14 | [Running an experiment](lessons/14-running-an-experiment.md) | A rule that separates a real gain from a lucky seed, and its false-alarm rate |
| 15 | [Rotary positions](lessons/15-rotary-positions.md) | A model with no table of places, which runs on text longer than it trained on |
| 16 | [Today's parts](lessons/16-todays-parts.md) | Llama's parts, giving a tiny reference Llama's scores |
| 17 | [What it costs](lessons/17-what-it-costs.md) | A model's memory and arithmetic counted before it runs, and this Mac's speed |
| 18 | [Data](lessons/18-data.md) | 210 MB of stories cleaned, split and turned into 52 million tokens on disk |
| 19 | [How big should it be?](lessons/19-how-big-should-it-be.md) | A law fitted to short runs, tested on runs it had not seen, and a size for the story model |
| 20 | [Many workers](lessons/20-many-workers.md) | Four processes that train one model and end with one process's weights |
| 21 | [The real run](lessons/21-the-real-run.md) | A model of 5 million parameters that writes short stories |
| 22 | [Sampling](lessons/22-sampling.md) | Control over how each token is picked, with what each setting does to twenty stories |
| 23 | [The KV cache](lessons/23-the-kv-cache.md) | A model that reads each token once while it writes |
| 24 | [Evaluation](lessons/24-evaluation.md) | A score with a range, a comparison with a verdict, and a task with a pass rate |

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
| GPT, with momentum | 11 | 2.564 |
| GPT, with AdamW | 11 | 2.372 |
| MLP, eight characters back, with AdamW | 11 | 2.551 |
| MLP, 64 characters back, with AdamW | 11 | 2.863 |
| GPT, with the loop, 5,000 steps | 12 | 2.206 |
| GPT, rotary positions | 15 | 2.210 |
| GPT, today's parts | 16 | 2.209 |

With plain gradient descent the GPT trails the MLP. Lesson 11 changes only the optimizer, and the GPT leads. Given that optimizer and as many guesses, the MLP does not catch it, and reading 64 characters it does worse than reading 8.

From lesson 18 the text is 210 MB of short stories, and the scoreboard starts again. The two cannot be compared: the stories are simpler text than Shakespeare.

| Model | Lesson | Bits per byte on unseen stories |
|---|---|---|
| Story model, 5,374,464 parameters | 21 | 0.575 |

## The same prompt at each stage

```
lesson 1    Once upon a timeron: Foleithathavout: Whe irtthogor s.
lesson 4    Once upon a time it plain? till I to susing to your me that was pifis last q
lesson 10   Once upon a timer deat tilses te grin and frit proner bese aff felling Eved.
lesson 12   Once upon a time of men, in backling all an embrace As your o'er peace will keep again the sword,
lesson 16   Once upon a time but fortune and 'I, She is my return waves; and play twice king;
lesson 21   Once upon a time, there was a little girl named Lily. She had a big red bag. Lily liked her bag very much.
```

## Run it

```bash
uv sync
uv run pytest
```

Each lesson has a page in `lessons/`, a short file in `gpt/` and a test in `tests/`. A lesson's numbers come from its own script, for example `uv run python -m gpt.bigram`, and are saved in `results/`.

From lesson 18 on, the scripts need the stories. `uv run python -m gpt.data` downloads 210 MB and prepares them, and `uv run python -m gpt.story` trains the story model. By lesson 17's timing that takes a little over two hours on an M4 laptop's graphics chip.

Two tests need outside downloads, and `uv run pytest` leaves them out. They need the `reference` extra, and the second needs GPT-2's weights (see lesson 10):

```bash
uv run --extra reference pytest -m reference
```

## How to practise

Open a file in `gpt/`, delete the body of one function, write it again from memory, and run its test.

## License

Apache-2.0. `data/tiny_shakespeare.txt` is public-domain text.
