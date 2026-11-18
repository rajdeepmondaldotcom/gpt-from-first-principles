# 41. Train on a GPU

By the end of this page the code that trained on a laptop has trained a bigger story model and a stronger chess player on a rented graphics card, and you know what each run cost, on one card and on four.

Every run so far used one laptop: a storyteller of 5,374,464 parameters and a chess player of 4,488,448, from one file of model code. We have not run that code on hardware built for it.

## The idea

We rent a machine with one NVIDIA L40S, at $1.861 an hour in November 2026. It is a graphics card: a graphics chip on its own board, with its own memory.

The model file, the training loop and the checks do not change. PyTorch calls an NVIDIA card `cuda` where it called the laptop's chip `mps`. Three things change.

Lesson 12's switch makes the numbers 16 bits wide.

There is more text: all 2,717,495 stories of TinyStories' training file, where lesson 18 took the first 210 MB, and four months of chess games, where lesson 38 took one. We score both story models on lesson 18's unseen stories, which neither has read.

The script sets the budget. It times a step of a model of 25 million parameters and counts what an hour of such steps computes: 5.0 × 10^16 multiplies and adds, 19 times lesson 21's budget. Chess gets the same.

For that budget lesson 19's law expects 0.813 nats at best, from 81 million parameters. The script takes the smallest size within 0.177 of that, the law's miss on its own runs: 14,555,904 parameters, at 0.923. Lesson 19's yardstick was tighter, 0.098, its largest miss on runs it had not seen, and it picks 25,170,944. We saw the difference after the run, so the script then trained each bigger size too.

We doubt the predictions themselves: at 34 times the largest budget in its fit the law was 0.263 nats too low for lesson 21's run, and this run spends 628 times. So the script also refits the law with that run added.

Then we try four cards. Lesson 20's four workers ended with the weights one process reaches, but they shared a laptop's cores and ran no faster. Here each has its own card and there is no coordinator: each worker draws the same batch from the same seed, takes its share, and sums its slopes with the others'.

## By hand

We could not rent a machine with four L40S cards that day, so the four-card test uses NVIDIA L4s, a slower card: $4.602 an hour for four and $0.805 for one.

4.602 / 0.805 = 5.7

To train a token for the same money, four cards must be 5.7 times as fast as one.

## The code

The run is lesson 21's `long_run` with the measured budget and `half=True`, lesson 12's switch. The new code is the worker of the four-card test:

```python
# gpt/gpu.py
    torch.manual_seed(0)  # the same seed: each worker starts from the same weights
    model = GPT(**shape).to(device)
    optimizer = AdamW(model.parameters(), rate=1e-3)
    # ...
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

`rank` is the worker's number, from 0, and `size` its count of windows. `x[rank::cards]` takes every `cards`-th window from window `rank` on: shares that do not overlap and together make the batch. `all_reduce`, from `torch.distributed`, is the exchange: each worker gives its tensor and gets back the sum of everyone's. Equal shares make that sum, divided by the number of cards, the whole batch's slope.

## The proof

```bash
uv run pytest tests/test_gpu.py
```

- The timer keeps the fastest step after the first, and the 16-bit switch is on when asked.
- The bigger pile keeps the unseen documents out and writes one end token for each one kept.
- The pick is the smallest size within the yardstick of the best.
- After four steps on the processor, two workers that share slopes match one worker on the whole batch to 0.00001.

The runs need an AWS account. `start.sh` takes a kind of machine and a limit in minutes, after which the machine and its disk are destroyed, and prints the machine's address, the `<host>` below. `run.sh` runs a command there and brings the results back.

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
 25,170,944 parameters,  37,501 steps, 0.6 times through the text: predicted 0.862 nats
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
the rules picked 88 of its 21,783 moves
the bigger model: rated +775 with the random mover at 0, 95% interval +714 to +829. The laptop's model: +355
this script, start to end: 239 minutes of one card, $7.41
```

Part of what it prints is above, and "19 laptops" means 19 times lesson 21's budget. Timings are from rented machines and move from run to run.

The card takes lesson 21's step in 34 ms where the laptop took 928. The bigger story model ends at 1.270 nats on unseen stories, 0.467 bits per byte, where lesson 21's had 1.565 and 0.575. Its run took 88 minutes, not 60: the hour was paced by a wider model's fastest step, and the run also tries three rates and checks its loss each 500 steps. Both laws predicted too low a loss, lesson 19's by 0.347 nats and the refit by 0.209.

The chess model is the pick's shape with lesson 38's 636 tokens. It makes a legal move at 987 of this script's 1,000 unseen positions, where the laptop's makes 932, and the rules pick 0.4% of its moves where they picked 3.6% of the laptop's. It takes 0.86 of the points from the laptop's model, and 0.89 from the grabber, which took 0.84 from the laptop's model. It is rated +775. The laptop's model is +355 in this fit and was +421 in lesson 39's: a rating depends on who else is in the fit.

On a machine with four cards, the same command runs the four-card test:

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

Four cards take the pick's step in 240 ms where one takes 1,004: 4.19 times as fast. The gain beats four because each of the four, exchange included, takes 7.5 ms a window on its 32, where one card takes 7.8 ms a window on 128. After 12 steps the two sets of weights differ by 0.000002 on average, where a weight moved 0.004. The largest gap, 0.003239, is far above the 0.00001 that lesson 20's test allows. The cards add the same numbers in another order, which may be why. A million tokens cost $0.0068 on one card and $0.0093 on four.

```bash
AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/stop.sh
```

## What it still cannot do

The bigger story model begins a story:

```
Once upon a time, there was a little girl named Lily. She had a pretty dress that she loved to wear. One day, she felt sad because she could not find her dress. She looked everywhere, but it was gone. Lily started to weep.
```

Only our own engine can serve the bigger story model, and lesson 37 found that 0.58 ms of each 1.29 ms pass of that engine is the fixed cost of its calls.

## Check yourself

Predict before you run: four L4 cards cost 5.7 times what one costs. Will a million tokens of training cost less on four?

<details><summary>Answer</summary>

No. Four cards are 4.19 times as fast, so a token costs 5.7 / 4.19 = 1.4 times as much. They buy time.
</details>

<details><summary>From lesson 18: why must the stories it held out stay out of the bigger pile too?</summary>

A story's own fingerprint decides its side, so cleaning the whole file puts the same stories on the unseen side. Had we trained on them, we would score the bigger model on text it had read.
</details>

Go deeper:

- PyTorch, [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html). PyTorch's own version, which sends the slopes in a few large pieces.
