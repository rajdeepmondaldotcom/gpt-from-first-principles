# GPT from first principles

Build a GPT yourself. Train it, serve it, teach the same code chess, then run it all on a rented graphics card.

You start with a loop that counts characters. Forty-five lessons later you have built each of these, and measured it:

| You build | Measured |
|---|---|
| A GPT, from its tokenizer up | It runs OpenAI's GPT-2 weights. On a test prompt it picks the token Hugging Face's code picks at each place, and the 31 scores we compare agree to within 0.0001 (lesson 10) |
| A story model, trained on a laptop | 0.575 bits per byte on unseen stories (lesson 21). On a rented card, 88 minutes and $2.74 bring that to 0.467 (lesson 41) |
| A storyteller that takes an order | Given a name and three words, by lesson 29 it gets all four into 35 of 200 stories, with a 95% interval of 0.13 to 0.23, where the story model got 0. A model trained on each story's score alone reaches 56, and leaves 59 of its stories without an end (lessons 25 to 29) |
| A server for it | With 16 requests inside at a time, 21.3 requests and 2,131 tokens a second from the laptop (lesson 36). On the card, vLLM keeps up with 128 requests a second: it answers them as fast as they arrive, and at 256 they wait. A million tokens written cost $0.065 there (lesson 42) |
| A chess player, from the same model code | An Elo rating of 1,785 on Stockfish's scale, which is fitted to a list of rated chess programs and is not Lichess's, after 12 hours of a card on 19 million games between strong players (lessons 44 and 45) |
| A comparison with an open model about 520 times the size | On 200 unseen children's stories, the one kind of text ours was trained on, ours scores 0.426 bits per byte and Qwen2.5-7B-Instruct, a general chat model, 0.514 (lesson 42). The number measures fit to this one kind of text and says nothing of what else each model can do |

A script in this repository prints each number and saves it in `results/`, and a test checks each block of output on each page against that file. Each page says where something did not work: [the list is below](#what-did-not-pay-off). A short way to judge the work: read [`tests/test_lessons.py`](tests/test_lessons.py), then lessons [14](lessons/14-running-an-experiment.md), [35](lessons/35-speculative-decoding.md), [41](lessons/41-train-on-a-gpu.md) and [42](lessons/42-serve-on-a-gpu.md).

Each lesson adds one idea, in code short enough to read in a sitting, with a test that checks it. You need Python and school maths. Lessons 41, 42, 44 and 45 also need an AWS account and about $60, near what they cost us. It is hard on purpose: each page has you work a sum by hand and predict a result before you run its script.

The model answers one prompt, `Once upon a time`, at seven stages. Lessons 1 to 16 learn from Shakespeare, and lessons 21 and 41 from children's stories. We show line breaks as spaces.

```
lesson 1    Once upon a timeron: Foleithathavout: Whe irtthogor s.
lesson 4    Once upon a time it plain? till I to susing to your me that was pifis last q
lesson 10   Once upon a timer deat tilses te grin and frit proner bese aff felling Eved.
lesson 12   Once upon a time of men, in backling all an embrace As your o'er peace will keep again the sword,
lesson 16   Once upon a time but fortune and 'I, She is my return waves; and play twice king;
lesson 21   Once upon a time, there was a little girl named Lily. She had a big red bag. Lily liked her bag very much.
lesson 41   Once upon a time, there was a little girl named Lily. She had a pretty dress that she loved to wear.
```

From lesson 25 the model takes an order. Asked for a story with Mira, a cat, the moon and the word brave, the storyteller that lesson 36 serves begins:

```
Once upon a time there was a brave girl named Mira. She was always looking for something fun, so she wanted to buy a big, beautiful fairy. One day she saw a beautiful, beautiful rainbow in the sky. She wanted to pick up the moon in her garden and take it home.
```

The whole story has Mira, the moon and brave, and no cat. On 200 other orders, each with its own name and three words, this model gets all four into 35 stories.

Lesson 38 trains lesson 21's code, unchanged, on records of chess games. Left to write a game by itself, it makes 26 legal moves, 13 for each side, before its first illegal one:

```
e4 e5 Nf3 Nc6 Bc4 Nf6 d3 h6 O-O Bc5 a3 a6 b4 Ba7 c3 d6 d4 exd4 cxd4 Bg4 Be3 O-O Qc2 Rfe8 h3 Bh5
```

*A note from Rajdeep.* I built this course to teach myself how a GPT is built, trained and served, and I am sharing it in case it helps you. The results on each page come from my own runs, and where something I tried did not help, the page says so. The course gives you the ideas behind training and inference work, each one built and measured. It does not give you a training run across many machines, a kernel you wrote for a graphics card, or a service kept up for real users. [MAP.md](MAP.md) and [PRACTICE.md](PRACTICE.md) say where to get those.

The numbers of lessons 1 to 40 and 43 come from an M4 MacBook, and those of lessons 41, 42, 44 and 45 from rented NVIDIA cards, which cost us $55.27 in all: $18.31 for lessons 41 and 42, and $36.96 for lessons 44 and 45. The code and its tests also run on Linux, on the processor, except lesson 37's check against Apple's engine, which needs a Mac.

Start with [lesson 0](lessons/00-the-whole-picture.md), which follows one request through the finished storyteller.

## The lessons

| # | Lesson | What you have at the end |
|---|---|---|
| 0 | [The whole picture](lessons/00-the-whole-picture.md) | One request followed through the finished storyteller, and a table of where each lesson sits |
| 1 | [Predict the next token](lessons/01-predict-the-next-token.md) | A program that writes text, and a number that says how good it is |
| 2 | [Gradients](lessons/02-gradients.md) | A model that teaches itself the table lesson 1 had to count |
| 3 | [Tensors](lessons/03-tensors.md) | The same model learning all of Shakespeare in seconds |
| 4 | [A neural language model](lessons/04-a-neural-language-model.md) | A model that reads eight characters back and writes things that look like words |
| 5 | [Tokens](lessons/05-tokens.md) | A tokenizer checked against OpenAI's, and a window that reaches more than twice as far |
| 6 | [Attention](lessons/06-attention.md) | Tokens that choose what to read, with one set of weights for any length |
| 7 | [Causal multi-head self-attention](lessons/07-causal-multi-head-self-attention.md) | A layer that reads the text in several ways at once, and trains on each place without seeing the answers |
| 8 | [Position](lessons/08-position.md) | A model that knows where each token stands |
| 9 | [The block](lessons/09-the-block.md) | A unit that still passes gradients back when stacked 24 deep |
| 10 | [The GPT](lessons/10-the-gpt.md) | One model that runs OpenAI's GPT-2 weights and, on a test prompt, picks the tokens Hugging Face's code picks |
| 11 | [Optimizers](lessons/11-optimizers.md) | The same GPT ahead of each earlier model, with only the rule for the step changed |
| 12 | [The training loop](lessons/12-the-training-loop.md) | A run you can stop and resume, bit for bit |
| 13 | [Is it working?](lessons/13-is-it-working.md) | Six checks that catch a silent bug, and a measured cure for a model that stores its text |
| 14 | [Running an experiment](lessons/14-running-an-experiment.md) | A rule that separates a real gain from a lucky seed, and its false-alarm rate |
| 15 | [Rotary positions](lessons/15-rotary-positions.md) | A model with no table of places, which runs on text four times longer than it trained on, and reads it worse: 3.452 bits per byte at 256 tokens against 2.210 at 64 |
| 16 | [Today's parts](lessons/16-todays-parts.md) | Llama's parts, giving a tiny reference Llama's scores |
| 17 | [What it costs](lessons/17-what-it-costs.md) | A model's memory and arithmetic counted before it runs, and this Mac's speed: 326 billion multiplies and adds a second (each multiply and each add counts as one: the field's FLOPs) |
| 18 | [Data](lessons/18-data.md) | 210 MB of stories cleaned, split and turned into 52 million tokens on disk |
| 19 | [How big should it be?](lessons/19-how-big-should-it-be.md) | A law fitted to short runs, tested on runs it had not seen, and a size for the story model |
| 20 | [Many workers](lessons/20-many-workers.md) | Four processes that train one model and end with one process's weights |
| 21 | [The real run](lessons/21-the-real-run.md) | A model of 5.4 million parameters that writes short stories: 0.575 bits per byte on unseen ones |
| 22 | [Sampling](lessons/22-sampling.md) | Control over how each token is picked, with what each setting does to twenty stories |
| 23 | [The KV cache](lessons/23-the-kv-cache.md) | A model that reads each token once while it writes |
| 24 | [Evaluation](lessons/24-evaluation.md) | A score with a range, a comparison with a verdict, and a task with a pass rate |
| 25 | [Fine-tuning](lessons/25-fine-tuning.md) | A storyteller that takes an order, a name and three words, and stops by itself: it passes 27 of 200 orders, where the story model passed 0 |
| 26 | [LoRA](lessons/26-lora.md) | 43,008 trained numbers that learn part of lesson 25's skill, with the old weights untouched |
| 27 | [Learning from a score](lessons/27-learning-from-a-score.md) | A model trained with no right answers, only a number for each story it wrote |
| 28 | [A model of what people prefer](lessons/28-a-model-of-what-people-prefer.md) | A scorer learned from pairs of stories, and a measurement of what training hard against it does |
| 29 | [Preferences without a scorer](lessons/29-preferences-without-a-scorer.md) | The storyteller trained on the pairs directly, with no second model |
| 30 | [An unwanted behaviour, measured](lessons/30-an-unwanted-behaviour-measured.md) | A rule the model keeps, its mistakes counted in both directions, and how soon more training loses it |
| 31 | [Latency and throughput](lessons/31-latency-and-throughput.md) | The waits a listener feels and the work a machine gets done, as listeners arrive faster and faster |
| 32 | [Batching](lessons/32-batching.md) | Many listeners sharing each pass, each still getting the story they would get alone |
| 33 | [Prefix caching](lessons/33-prefix-caching.md) | A beginning that many prompts share, read once and kept |
| 34 | [Quantization](lessons/34-quantization.md) | Weights in 8 bits or 4, with what each costs in memory, loss and speed |
| 35 | [Speculative decoding](lessons/35-speculative-decoding.md) | A small model guessing four tokens ahead, with the big model's chances kept exactly, and no faster on this laptop: 480 tokens a second against 490 |
| 36 | [The server](lessons/36-the-server.md) | The storyteller in a browser: streamed, read aloud, and serving other listeners at the same moment, at 21.3 requests and 2,131 tokens a second |
| 37 | [A real engine](lessons/37-a-real-engine.md) | The model in Apple's MLX engine with the scores it has in ours, and both engines timed |
| 38 | [The same code, a different world](lessons/38-the-same-code-a-different-world.md) | Lesson 21's model code, unchanged, trained from nothing on records of chess games, with its legal moves counted |
| 39 | [Can it play?](lessons/39-can-it-play.md) | Whole games against three opponents, and an Elo rating with a range, on a scale of our own: +421 |
| 40 | [The board in its head](lessons/40-the-board-in-its-head.md) | A chess position read out of the model's inner numbers, and the chance of a move changed by editing them |
| 41 | [Train on a GPU](lessons/41-train-on-a-gpu.md) | The same code on a rented graphics card: a bigger story model, a stronger chess player, four cards sharing a step, and each run's cost in dollars |
| 42 | [Serve on a GPU](lessons/42-serve-on-a-gpu.md) | The bigger story model served by vLLM, the request rate it keeps up with, and the price of a million tokens: $0.065 for our model, and $0.155 for one about 520 times its size |
| 43 | [Split the model](lessons/43-split-the-model.md) | One layer's weights and the optimizer's averages shared among four workers, with each sum checked against the model in one piece |
| 44 | [A stronger player](lessons/44-a-stronger-player.md) | A chess model of 76 million parameters, trained for 12 hours of a card on 19 million games between strong players, with each choice in the run and its cost: $22.33 for the 12 hours of steps, about $30 on the clock |
| 45 | [How strong is it?](lessons/45-how-strong-is-it.md) | 1,400 games against Stockfish held to known ratings, 1,300 of them by the new player, and its Elo rating on that scale: 1,785, with a range of 1,737 to 1,836 |

Four guides go with the lessons.

- [MAP.md](MAP.md): what the lessons leave out, and the open resource we would read first for each thing.
- [PRACTICE.md](PRACTICE.md): nine routines for practice the lessons cannot give, most of them for what 11 job postings at AI labs ask for.
- [GLOSSARY.md](GLOSSARY.md): each term with the lesson that defines it, and our plain words beside the field's.
- [PROCESS.md](PROCESS.md): how each lesson is made and checked, so you can check any claim yourself.

## The scoreboard

We score each model on the same unseen tenth of Shakespeare, in bits per byte. Lower is better.

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

With plain gradient descent the GPT trails the MLP. Lesson 11 changes only the optimizer, and the GPT leads. With the same optimizer, the same steps and as many predictions, the MLP still scores worse, and reading 64 characters it does worse than reading 8.

From lesson 18 the text is 210 MB of short stories, and the scoreboard starts again. You cannot compare the two: the stories are simpler text than Shakespeare.

| Model | Lesson | Bits per byte on unseen stories |
|---|---|---|
| Story model, 5,374,464 parameters | 21 | 0.575 |
| Story model, 14,555,904 parameters, 88 minutes of a rented card | 41 | 0.467 |

Lesson 24 reads each unseen story alone, and lesson 21's model scores 1.480 nats a token there, with a 95% interval of 1.463 to 1.497.

From lesson 25 we ask how often the model does what it is asked, in place of how well it predicts text. Each row is 200 orders made from unseen stories, and a story passes when it has the asked name and all three asked words.

| Model | Lesson | Orders passed, of 200 |
|---|---|---|
| Story model | 25 | 0 |
| Fine-tuned on 60,404 examples | 25 | 27 |
| LoRA, rank 8 beside queries and values | 26 | 10 |
| Lesson 25's model, stories drawn afresh | 27 | 30 |
| Kept what passed, and fine-tuned on it | 27 | 37 |
| Learned from each story's score | 27 | 56 |
| Learned from pairs, with no scorer | 29 | 35 |

The server of lesson 36 loads the last of these: lesson 28's stand-in readers give its stories 0.62 on their scale of 0 to 1, where they gave lesson 25's 0.53. The model that passes 56 leaves 59 of its 200 stories without an end.

Lessons 31 to 37 measure the model as a service, on one M4 laptop. Lesson 36 serves the storyteller, and the others time lesson 21's story model, which has the same shape. Timings move from run to run.

| Measure | Lesson | Result |
|---|---|---|
| First token for one listener, after a prompt of 192 tokens | 31 | 8 ms |
| Tokens a second for one listener | 31 | 539 |
| Tokens a second that memory alone would allow | 31 | 4,231 |
| Median wait for a first token at 15.1 requests a second, one request at a time | 31 | 1,737 ms |
| The same, with room for 16 requests in each pass | 32 | 8 ms |
| Tokens a second in all, 32 listeners sharing each pass | 32 | 2,275 |
| The weights at 8 bits | 34 | 8.6 MB in place of 21.5, for 0.0001 nats a token more loss, and slower writing in our engine |
| Requests a second over the web, 16 at a time | 36 | 21.3 |
| Tokens a second in all, in the same test | 36 | 2,131 |
| The fixed cost of 516 calls into PyTorch in one token's pass on the processor, worked out from two models' timings | 37 | 0.58 ms of 1.29 ms |
| Tokens a second for 16 listeners on the processor: our engine, then Apple's MLX with the same weights | 37 | 4,134, then 5,943 |

We score the chess model of lesson 38 by the rules of the game and by play. Points are the share of the points it took: a win is 1 and a draw a half.

| Measure | Lesson | Result |
|---|---|---|
| Legal moves at 1,000 positions from unseen games | 38 | 929. At the 808 positions no training game reaches, 737 |
| Points against a player that moves at random, 200 games | 39 | 0.92 |
| Points against a player that looks one move ahead, 200 games | 39 | 0.16 |
| Points against itself at step 2,500, 200 games | 39 | 0.64 |
| Squares a linear probe names from the model's inner numbers | 40 | 80.7% after block 5, against 67.3% for naming each square's most usual piece |

Lessons 41 and 42 run the same code on rented NVIDIA cards: one L40S, except for the four-card timing, which ran on four L4s.

| Measure | Lesson | Result |
|---|---|---|
| One training step of lesson 21's model: the laptop, the card, the card with 16-bit numbers | 41 | 928 ms, 49 ms, 34 ms |
| A story model of 14,555,904 parameters, 88 minutes of the card, $2.74 | 41 | 1.270 nats a token on unseen stories (0.467 bits per byte), where the laptop's model had 1.565 (0.575) and lesson 19's law expected 0.923 nats |
| The same budget on 25,170,944 parameters, then on 47,678,720 | 41 | 1.241 nats, then 1.271: the size lesson 19's own rule would have picked was the best of the three |
| A chess model of 13,226,880 parameters: legal moves at 1,000 unseen positions, and its points against the laptop's model | 41 | 987 (the laptop's: 932 at the same positions), and 0.86 |
| One step on four cards, against the same step on one | 41 | 4.19 times as fast, on a machine that costs 5.7 times as much an hour: 1.4 times the price for each token |
| Tokens a second for one listener: our engine on the card, then vLLM with the same weights | 42 | 110, then 786 |
| Requests a second vLLM keeps up with, and the price of a million tokens there | 42 | 128, at $0.065. At 256 and at 512 it writes about 12,800 tokens a second and requests wait |
| Qwen2.5-7B-Instruct, 7.6 billion parameters, one listener: tokens a second the card's memory allows, then what it gets | 42 | At 16 bits: 52, then 50. At 8 bits: 97, then 73 |
| 200 unseen stories, each scored from its own start, in bits per byte: our 14.6 million parameters, then Qwen2.5-7B-Instruct's 7.6 billion | 42 | 0.426, then 0.514. A million tokens cost $0.065, then $0.155 |

Lesson 43 is back on the laptop. Its four workers are in one process, so it counts bytes and checks sums, and times nothing.

| Measure | Lesson | Result |
|---|---|---|
| What one of four workers holds to train lesson 41's story model: a whole copy, then with AdamW's averages shared out | 43 | 232.9 MB, then 155.0 MB for the fullest worker |
| What the workers add up between them in a step: their slopes, then their parts of eight split MLPs | 43 | 58.2 MB, then 201.3 MB |

Lessons 44 and 45 are on the rented L40S again: 12 hours of training steps on games between strong players, then 1,400 games against Stockfish, a chess program that can be held to a rating. Lesson 41's model plays 100 of them. The new player plays 500 to choose what we tell it about a game, 700 for its rating, and 100 with Stockfish on a slower clock.

| Measure | Lesson | Result |
|---|---|---|
| A chess model of 76,286,080 parameters, after 443,470 steps on 19,219,755 games between players rated 2,200 or more | 44 | 1.447 nats a token on unseen games, where a token is a move. The 12 hours of steps cost $22.33, and the stage took 16 hours on the clock, about $30 |
| One training step with lesson 6's attention, then with PyTorch's own routine in its place | 44 | 257 ms, then 97 ms |
| Tokens a second through a training step of 32 windows, then of 128 | 44 | 84,095, then 63,206 |
| Points against Stockfish held to 1,320, for lesson 41's chess model. Then against Stockfish held to 1,400, for the new one | 45 | 0.23, then 0.80 |
| The new player's Elo rating on Stockfish's scale, fitted to 700 games against seven settings from 1,400 to 2,600 | 45 | 1,785, with a range of 1,737 to 1,836. Match by match the rating climbs from 1,641 against the weakest setting to 2,122 against the strongest, so one number does not suit them all |
| Points against Stockfish held to 1,800, when it has 0.05 seconds a move and when it has 0.5 | 45 | 0.36, then 0.41 |

The table below lists the chess players of the course, each with its Elo rating. The first four are on a scale of our own, and the last two on Stockfish's.

| Player | Lesson | Elo rating | Scale |
|---|---|---|---|
| A player that moves at random | 39 | 0 | Ours: it is held at 0 |
| The laptop's chess model | 39 | +421, with a range of +378 to +465 | Ours |
| A player that looks one move ahead | 39 | +738, with a range of +690 to +798 | Ours |
| The bigger chess model from the rented card | 41 | +775, with a range of +714 to +829 | Ours. In this fit the laptop's model is +355 |
| The same model against Stockfish | 45 | About 1,110 | Stockfish's, from one match: 0.23 of the points against its weakest setting, 1,320. That is below the scale, so no setting can check it |
| The stronger player | 45 | 1,785, with a range of 1,737 to 1,836 | Stockfish's, from 700 games against seven settings |

## What did not pay off

Each of these is on its lesson's page, with the run that showed it.

| What we tried | Lesson | What happened |
|---|---|---|
| A model with no table of places, on text longer than it trained on | 15 | It runs, and reads worse: 3.452 bits per byte at 256 tokens against 2.210 at 64 |
| Four workers sharing a step on one laptop | 20 | The weights one process reaches, and no faster: 126 ms a step against 115 |
| Weights at 8 bits | 34 | 8.6 MB in place of 21.5, and slower writing in our engine |
| A small model guessing ahead for the big one | 35 | No faster: 475 and 480 tokens a second with two drafts, against 490 without |
| A law fitted on small runs, to size a run 628 times larger | 41 | 0.347 nats too low, and the size it led us to was not the best of the three we then trained |
| Four graphics cards in place of one | 41 | 4.19 times as fast, on a machine that AWS prices at 5.7 times the one-card machine, so a token costs 1.4 times as much. The step itself sped up by more than four |
| A chess player as strong as the players it learned from, who were rated 2,200 or more on Lichess | 45 | An Elo rating of 1,785 on Stockfish's scale, which is not Lichess's, and 0.19 of the points from Stockfish held to 2,200 |

## Run it

```bash
uv sync
```

```bash
uv run pytest
```

Each lesson has a page in `lessons/`, code in a file in `gpt/` (most are under 200 lines, the longest is under 500, and some serve two or three lessons) and tests in `tests/`. Most lessons' numbers come from that file's script, for example `uv run python -m gpt.bigram`, and are saved in `results/`.

From lesson 18 on, the scripts need the stories and a trained model. `uv run python -m gpt.data` downloads 210 MB and prepares them, and `uv run python -m gpt.story` trains the story model. By lesson 17's timing that takes 2.3 hours on an M4 laptop's graphics chip. `gpt.finetune` (lesson 25) makes the storyteller. The server loads the storyteller as lesson 29 left it, so first run `gpt.finetune`, `gpt.reinforce` and `gpt.prefer` in that order, or download the models below.

To hear a story, start the server and open `http://127.0.0.1:8000`:

```bash
uv run python -m gpt.server stay
```

Lessons 38 to 40, 44 and 45 need the `chess` extra: python-chess, which knows the rules, and readers for the two formats Lichess publishes its games in.

```bash
uv sync --extra chess
```

```bash
uv run python -m gpt.games
```

`gpt.games` downloads 129 MB of games, prepares them and trains the chess model for 10,420 steps. If the downloaded chess model is in place, it finds the finished run and goes straight to its counts.

Three tests check our code against outside code: OpenAI's tiktoken, Hugging Face's GPT-2 and Apple's MLX. `uv run pytest` leaves them out. They need the `reference` extra. The GPT-2 test also needs GPT-2's weights (see lesson 10), and the MLX test runs only on a Mac:

```bash
uv run --extra reference pytest -m reference
```

Every test runs on a laptop. Only the scripts of lessons 41, 42, 44 and 45 need a rented machine: an AWS account that may start G instances in us-east-1, and the `aws` command set up with a profile of yours. `aws/start.sh` starts a machine that destroys itself at a time limit, `aws/run.sh` runs a lesson on it and brings the results back, `aws/cost.sh` lists what the machines still running have cost, and `aws/stop.sh` destroys each machine and lists any left. Each page gives its commands. With the downloaded `story_big`, lesson 42 needs no run of lesson 41. Lesson 44's games stage downloads 999 GB, one file at a time.

## The trained models

You can skip the long runs. The story model, the storyteller at two stages, the chess model at two stages, lesson 41's two bigger models and lesson 44's stronger player are on Hugging Face, each with the numbers we measured on it. Each command fills `data/cache` with the folders the scripts load: `story`, `storyteller`, `preferred` and `story_big` from the first, and `chess`, `chess_early`, `chess_big`, `strong`, the unseen games of lesson 44 and the chess run's last checkpoint from the second.

```bash
uvx --from huggingface_hub hf download rajdeepmondal/gpt-from-first-principles-storyteller --local-dir data/cache
```

```bash
uvx --from huggingface_hub hf download rajdeepmondal/gpt-from-first-principles-chess --local-dir data/cache
```

The scripts still need the text itself. `gpt.data` fetches and prepares the stories in minutes, and `gpt.games` does the same for the chess games.

## Sources and thanks

Each lesson ends with the sources it rests on, and [MAP.md](MAP.md) lists open material for everything else. We read four works before writing the first lesson.

- Sebastian Raschka, [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch). We took from it the habit of working one case by hand before the general one.
- Knowlytix, [Building Language Models from Scratch](https://github.com/knowlytix/llm-from-scratch). We used its order of topics as a checklist.
- Philip Kiely, [Inference Engineering](https://www.baseten.co/inference-engineering/). We read it as a map of serving.
- elizabetht, [100 days of LLM inference engineering](https://github.com/elizabetht/100-days-of-inference). It has experiments worth building again.

Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and Stanford's [CS336](https://cs336.stanford.edu/) showed what teaching from nothing, with a test for each part, can look like.

None of their text, code or figures is in this repository.

## License

Apache-2.0. `data/tiny_shakespeare.txt` is public-domain text. The stories are from TinyStories, under CDLA-Sharing-1.0, and the chess games from the Lichess open database, under CC0; neither is stored here. The chess lessons use the `python-chess` library, which is under GPL-3. It is an optional extra, and no code of it is in this repository. Lesson 42 runs vLLM and the Qwen2.5-7B-Instruct model, both under Apache-2.0, on a rented machine. Lesson 45 runs Stockfish there too, which is under GPL-3. None of the three is stored here or installed by `uv sync`.
