# 14. Running an experiment

By the end of this page you can say whether a change made a model better, and how often your rule for saying so is wrong.

Lesson 13 compared two runs, one of each setting. Lesson 12 took its schedule on trust. We cannot trust either result yet, because we do not know how far apart two runs of the same setting land.

## The idea

A run starts from random weights and draws random batches, and lesson 11's seed fixes both. On `cpu` the same seed gives the same run to the last bit, and a new seed gives a different one.

So run one setting on five seeds. Lesson 12's model, trained for 600 steps, ends between 1.896 and 1.943 nats on unseen text, with only the seed changed. A change that moves one run by 0.02 nats has shown nothing.

To compare two settings, run both on the same seeds and take the difference for each seed. Average the differences. Then ask how sure that average is: work out the spread of the differences as lesson 9 did for a vector, but divide by one less than their count, as By hand explains. Then divide the spread by the square root of the number of seeds. The result is called the standard error, and here we call it the error.

The rule: count a difference as real when it is more than 3 errors from zero. Five seeds give only a rough error, so we set the bar above the usual 2.

A rule like this is wrong at a rate we can measure. A test draws 10,000 pretend experiments in which nothing changed, so each difference is noise. With five seeds the rule passes 4 or 5 of each 100 as real. With three seeds it passes 9.

The rule also misses real changes. A change three times the spread of the differences is found more than 90 times in 100 with five seeds. A change as large as that spread is missed more often than found with five seeds, and found more than 80 times in 100 with twenty.

Three habits go with the rule:

- Change one thing at a time. If two things change, you cannot tell which one caused the difference.
- Search a rate by factors. Lesson 13's planted rate was wrong by a factor of 100. So try 0.0003, 0.001, 0.003, 0.01 and 0.03, each about three times the last, on a few seeds. This is called a sweep. Trust a rate that wins on each seed.
- Say what was measured. Our runs are 600 steps long. A rate that wins at 600 steps may lose at 5,000.

## By hand

Setting A scores 2.00 on each of five seeds. Setting B scores 2.02, 1.99, 2.03, 2.00 and 2.01.

| Step | Result |
|---|---|
| Differences, B minus A | 0.02, -0.01, 0.03, 0.00, 0.01 |
| Their mean | 0.01 |
| Each minus the mean, squared | 0.0001, 0.0004, 0.0004, 0.0001, 0 |
| Sum, divided by 4 | 0.00025 |
| Spread, its square root | 0.0158 |
| Error, the spread over the root of 5 | 0.0071 |
| Mean over error | 1.4 |

B looks worse by 0.01. That is 1.4 errors from zero, which is under 3, so the rule does not count B as worse.

We divide by 4 because the mean was worked out from these five numbers, which makes their spread look smaller than it is.

## The code

```python
# gpt/experiment.py
def trial(make, ids, val_ids, seed, steps, peak=3e-3, **loop):
    """One run of lesson 12's loop, which also gets `loop`. Returns the loss on unseen text."""
    torch.manual_seed(seed)  # the seed fixes the starting weights and the batches
    model = make()
    generator = torch.Generator().manual_seed(seed)
    optimizer = AdamW(model.parameters(), rate=0.0, decay=0.1)
    for _ in train(model, optimizer, ids, steps, peak=peak, generator=generator, **loop):
        pass
    return evaluate(model, val_ids)


def compare(a, b):
    """Results of two settings, run on the same seeds. Returns the mean of b - a, and its error."""
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    mean = sum(diffs) / len(diffs)
    spread = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    return mean, spread / math.sqrt(len(diffs))  # the error shrinks as the seeds grow
```

`make` is a function that builds the model, so each trial starts fresh. `**loop` collects any extra named settings and hands them to lesson 12's `train`. `zip(a, b, strict=True)` pairs the two lists and raises an error if one is longer. `rate=0.0` is a placeholder, because the loop sets the rate before each step. `val_ids` is the unseen text, and `evaluate` gives the mean loss on it.

## The proof

```bash
uv run pytest tests/test_experiment.py
```

- The hand example gives a mean of 0.01, an error of 0.0071 and a ratio of 1.41.
- The same seed gives the same loss, and a new seed a different one.
- With nothing changed, the rule passes between 3 and 6 experiments in 100 at five seeds, and between 8 and 11 at three.
- A planted change three times the noise is found. One as large as the noise needs twenty seeds.
- On a tiny model, a rate 100 times too low is more than 3 errors worse, and each of three seeds picks 0.01 or 0.03 out of five rates.

```bash
uv run python -m gpt.experiment
```

```
one setting, five seeds: 1.8962 1.9195 1.9256 1.9106 1.9433
rate 0.0003: 2.2854 2.2812 2.2763
rate 0.001 : 2.0598 2.0412 2.0598
rate 0.003 : 1.8962 1.9195 1.9256
rate 0.01  : 1.8580 1.8408 1.8470
rate 0.03  : 1.8895 1.8688 1.8737
best rate for each seed: [0.01, 0.01, 0.01]
the same setting on five new seeds: +0.0040 nats, error 0.0100, ratio 0.4
half the width: +0.1458 nats, error 0.0095, ratio 15.4
a fixed rate in place of the schedule: +0.2220 nats, error 0.0246, ratio 9.0
```

Each number is the loss on unseen text after 600 steps of lesson 12's model and loop.

Two of the comparisons are planted. The same setting on five new seeds differs by 0.4 errors, and the rule rejects it. Half the width is worse by 15.4 errors, and the rule passes it.

The other two were open questions. A fixed rate of 0.003 with no warmup is worse than lesson 12's schedule by 0.222 nats, 9.0 errors from zero. That is warmup and the falling rate together, at 600 steps. We cannot tell from it which of the two helps.

In the sweep, lesson 12's rate was too low for a run this short. A peak of 0.01 wins on each of the three seeds, by 0.065 nats on average over 0.003. Lessons 15 and 16 keep 0.003, so that their numbers can be set beside lesson 12's.

## What it still cannot do

Each of these models reads 64 tokens and no more. Hand one a 65th, and lesson 8's table of places has no row for it: the model stops with an error.

## Check yourself

Predict before you run: five seeds gave an error of 0.010 nats. About how many seeds would bring it to 0.005?

<details><summary>Answer</summary>

Twenty. The error falls with the square root of the number of seeds, so halving it takes four times as many.
</details>

<details><summary>You change the rate and the width together, and the loss improves by 10 errors. What have you learned?</summary>

That the pair of changes helps. It does not show which one helped, or whether one of them hurts.
</details>

<details><summary>With three seeds and nothing changed, how often does the rule believe a difference?</summary>

About 9 times in 100, twice as often as with five seeds.
</details>

<details><summary>From lesson 13: dropout and decay moved the loss on unseen text from 3.648 to 2.286 nats, one run of each. Could that be luck?</summary>

One run of each is thin evidence, and two things changed at once. Here five runs of one setting land within 0.05 nats of each other, and 1.36 is far more. But those five used another setup, so lesson 13's own noise is still unmeasured.
</details>

Go deeper:

- Google, [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook). How to plan a round of experiments, and what to sweep.
- Bouthillier and others, [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) (2021). How much of a reported gain is the seed.
- Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (2024). The standard error, applied to model evaluations.
