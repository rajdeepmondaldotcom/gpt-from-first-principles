# 41. Train on a GPU

By the end of this page the code that trained on a laptop has trained a bigger storyteller and a stronger chess player on a rented graphics card, and you know what each run cost and what four cards buy.

Every run so far used one laptop: a storyteller of 5,374,464 parameters and a chess player of 4,488,448, from one file of model code. We have not run that code on hardware built for it.

## The idea

We rent a machine with one NVIDIA L40S, a graphics card with 48 GB of memory of its own, at $1.861 an hour in November 2026.

The model file, the training loop and the checks do not change. PyTorch calls an NVIDIA card `cuda` where it called the laptop's chip `mps`. Three things do change.

The numbers are 16 bits wide, by lesson 12's switch. On this card a step of lesson 21's model takes 49 ms in 32 bits and 34 ms in 16.

There is more text: the whole of TinyStories, 2,717,495 stories, where lesson 18 took the first 210 MB, and four months of chess games where lesson 38 took one. The stories lesson 18 held out stay out, so both storytellers are scored on text neither has read.

The budget is measured: the script times a step of a mid-sized model and counts an hour at that pace. That is 5.0 × 10^16 multiplies and adds, 19 times the budget of lesson 21's run, and chess gets the same.

For that budget, lesson 19's law expects 0.813 nats at best, from 81 million parameters. It misses its own runs by 0.177, so it cannot tell that from the 0.923 it expects of 14,555,904 parameters, and lesson 19's rule takes the smallest shape the law cannot tell from the best.

We doubt the 0.923. At 34 times the largest budget in its fit, the law was 0.263 nats too low for lesson 21's run, and this run is at 628 times. So the script fits the law again with that run added, and prints both predictions.

Last, four cards. Lesson 20's four workers ended with one process's weights, but they shared a laptop's cores and none beat one process. Here each worker has a card of its own and there is no coordinator: each draws the same batch from the same seed, takes its own share, and the workers add their slopes up among themselves.

## By hand

The budget pays for 63,407 steps of the pick, each on 32 windows of 256 tokens:

63,407 × 32 × 256 = 519,430,144 tokens

The cleaned stories hold 551,449,289, so the run reads 0.94 of them once.

No machine with four L40S cards was free on the day, so the four are NVIDIA L4s, a slower card: $4.602 an hour for four and $0.805 for one.

4.602 / 0.805 = 5.7

To train a token for the same money, four cards must be 5.7 times as fast as one.

## The code

The run is lesson 21's `long_run` with the measured budget and `half=True`, lesson 12's switch for 16-bit numbers. The new code is the worker:

```python
# gpt/gpu.py
    torch.manual_seed(0)  # the same seed: each worker starts from the same weights
    model = GPT(**shape).to(device)
    optimizer = AdamW(model.parameters(), rate=1e-3)
    generator, times = torch.Generator().manual_seed(0), []
    for _ in range(steps):
        began = time.perf_counter()
        x, y = batch(ids, model.context, size * cards, generator)  # each worker draws one batch
        model.zero_grad()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=half):
            loss = loss_on(model, x[rank::cards], y[rank::cards])  # and takes its own share
        loss.backward()
        for p in model.parameters():
            distributed.all_reduce(p.grad)  # now each worker holds the sum of everyone's slopes
            p.grad /= cards
        optimizer.step()
```

`rank` is the worker's number, from 0, and `x[rank::cards]` is each `cards`-th window from that number on: shares that do not overlap and together make the batch. `all_reduce` is PyTorch's call for the exchange: each worker gives its tensor and gets back the sum of everyone's. The shares are equal, so that sum over the count of cards is the slope of the whole batch.

## The proof

```bash
uv run pytest tests/test_gpu.py
```

- A step's time is the fastest after the first, and the 16-bit switch is on when asked.
- The bigger pile keeps the unseen documents out and writes one end token for each one kept.
- The pick is the smallest shape the law cannot tell from the best.
- On the processor, two workers that share slopes end four steps where one worker on the whole batch does, to 0.00001.

The runs need an AWS account. Three scripts in `aws/` start a machine, run a command on it and destroy it, and a machine shuts itself down at the time limit you give it.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/start.sh g6e.xlarge 300
aws/run.sh <host> python -m gpt.gpu
```

```
the card: NVIDIA L40S
stories read: 2,717,495, kept: 2,717,461, written for training: 2,690,389, 551,449,289 tokens
the story model's training step: 928 ms on the laptop, 49 ms on this card, 34 ms with 16-bit numbers
an hour of this card: 5.0e+16 multiplies and adds, 19 laptops
 14,555,904 parameters,  63,407 steps, 0.9 times through the text: predicted 0.923 nats
 81,022,464 parameters,  11,925 steps, 0.2 times through the text: predicted 0.813 nats
the pick: 14,555,904 parameters, width 384, 8 blocks
step 500: train 2.512, unseen 2.582 nats
step 63407: train 1.260, unseen 1.269 nats
the bigger model, unseen stories: 1.270 nats a token, 0.467 bits per byte. The laptop's model: 1.565 and 0.575
lesson 19's law predicted 0.923 nats: off by +0.347
the law with lesson 21's run added predicted 1.062 nats: off by +0.209
the run: 88 minutes of one card, $2.74
the bigger chess model: 13,226,880 parameters, 69,103 steps, 1.2 times through the games, unseen 1.014 nats where the laptop's had 1.317
the laptop's model: 932 of 1,000 moves are legal, 95% interval 0.915 to 0.947
the bigger model: 987 of 1,000 moves are legal, 95% interval 0.978 to 0.993
the bigger model against the grabber: 0.89 of the points in 200 games, 95% interval 0.86 to 0.93
the bigger model against the laptop's model: 0.86 of the points in 200 games, 95% interval 0.83 to 0.90
the bigger model: rated +775 with the random mover at 0, 95% interval +714 to +829. The laptop's model: +355
this script, start to end: 239 minutes of one card, $7.41
```

The card takes lesson 21's step in 34 ms where the laptop took 928: 27 times as fast. The bigger storyteller ends at 1.270 nats on unseen stories, 0.467 bits per byte, where lesson 21's had 1.565 and 0.575. Its run took 88 minutes, not 60: the hour was timed on a wider model than the pick, and the run also tries three rates.

Both laws were too hopeful: lesson 19's by 0.347 nats, and the one that had seen lesson 21's run by 0.209. The law said a bigger model on more text would do better, and it did, by less than the law said. We trained one size, so we still cannot say the pick was the best one.

The chess model makes a legal move at 987 of 1,000 unseen positions, where the laptop's made 932. It takes 0.86 of the points from the laptop's model, and 0.89 from the grabber, which the laptop's model took 0.16 from. It is rated +775. The laptop's model is +355 in this fit and was +421 in lesson 39's: a rating depends on who else is in the fit.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/start.sh g6.12xlarge 60
aws/run.sh <host> python -m gpt.gpu
```

```
one card, 128 windows a step: 1004 ms, 32,644 tokens a second
4 cards, 32 windows each: 240 ms a step, 136,741 tokens a second, 4.19 times as fast
12 steps of 32 windows in 32-bit numbers: a weight moved 0.004186 on average. One card's weights and 4 cards' differ by 0.000002 on average, 0.003239 at most
a million tokens of training: $0.0068 on one card, $0.0093 on 4
```

Four cards take the step in 240 ms where one takes 1,004: 4.19 times as fast. That is over four because one card spends 7.8 ms on each of 128 windows and 7.5 ms on each of 32, and the exchange of slopes is too small beside either to show. After 12 steps the two sets of weights differ by 0.000002 on average, where a weight moved 0.004. The largest gap, 0.003239, is far above lesson 20's 0.00001: the cards add the same numbers in another order, and over 12 steps a difference in the last bits grows at a few weights. A million tokens cost $0.0068 on one card and $0.0093 on four: the four bought time, and charged for it.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/stop.sh
```

## What it still cannot do

The bigger model begins a story:

```
Once upon a time, there was a little girl named Lily. She had a pretty dress that she loved to wear. One day, she felt sad because she could not find her dress. She looked everywhere, but it was gone. Lily started to weep.
```

Nothing serves it yet. The one engine we have is our own, and lesson 37 found that 0.58 of each 1.29 ms pass of it is the fixed cost of its calls.

## Check yourself

Predict before you run: lesson 19's law expects 0.923 nats from this run, and it was 0.263 too low for lesson 21's. Will this run end above 0.923 or below it?

<details><summary>Answer</summary>

Above: 1.270 nats, 0.347 over. The law was fitted to runs of a few minutes, and this one spends 628 times the largest of their budgets.
</details>

<details><summary>From lesson 18: why must the stories it held out stay out of the bigger pile too?</summary>

A story's own fingerprint decides its side, so cleaning the whole file puts the same stories on the unseen side. Had they been trained on, the bigger model would be scored on text it had read.
</details>

Go deeper:

- PyTorch, [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html). PyTorch's own version: the slopes go in a few large pieces while the pass back still runs.
