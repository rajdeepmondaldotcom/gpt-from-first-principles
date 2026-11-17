# 29. Preferences without a scorer

By the end of this page the storyteller has learned from lesson 28's pairs with no second model in between.

The scorer of lesson 28 is a second model to train and keep, and training against it drove the readers' number to 0.05. The pairs themselves already say which story is better, and we have not learned from them directly.

## The idea

A pair says that for one order, story A is better than story B. Training should then make A more likely and B less likely. Our model gives each text a chance, so "more likely" has an exact meaning: the log of A's chance should rise and the log of B's should fall.

We cannot compare the two log chances as they stand. They differ for reasons that say nothing about which story is better: each token lowers a story's log chance, so length alone can put one story far below another. So we compare how far each has moved since training began. Keep a frozen copy of the model, made before this training starts, and call it the reference. A long story is unlikely under both, and the difference between them cancels its length.

For each story, ask how much more likely our model makes it than the reference does: the log of its chance under our model, minus the log of its chance under the reference. Call that the story's gain. Then for a pair,

lead = gain of the better story - gain of the worse story

We want the lead to grow, and the loss is lesson 28's bet again, with the lead times a small number beta in the place of x.

1. At the start our model is the reference. Each gain is 0, each lead is 0, and the loss is ln 2.
2. The loss falls as the better story gains on the worse one. A rise in the better and a fall in the worse count alike.
3. A pair with a large lead has little left to teach: its loss is nearly flat there, so it pushes little.
4. Beta sets how large a lead counts as large. A lead is in nats, summed over a story, and our stories run to about 180 tokens. With our beta of 0.1, a lead of 10 nats, under 0.06 a token, makes x = 1 and a bet of 0.73.

This is called direct preference optimization, or DPO. It needs no scorer and no stories written during training. The pairs are fixed, and the reference's part of each lead can be worked out once.

Its authors derive the loss from lesson 28's setting with one addition, a penalty for drifting from the reference. Let a story's score be beta times its gain, and the storyteller is its own scorer: the two models of lesson 28 fold into one.

## By hand

In one pair, each log chance is a sum over a story's tokens.

| | Better story | Worse story |
|---|---|---|
| Our model | -1.0 | -6.0 |
| The reference | -2.0 | -3.0 |
| Gain | +1.0 | -3.0 |

The lead is 1.0 - (-3.0) = 4.0. With a beta of 0.1, x is 0.4.

| Step | Sum | Value |
|---|---|---|
| Chance the bet puts on the better story | 1 / (1 + e^-0.4) | 0.599 |
| Loss | -ln 0.599 | 0.513 |

## The code

```python
# gpt/prefer.py
def preference_loss(ours_better, ours_worse, frozen_better, frozen_worse, beta=0.1):
    """DPO. How much more we favour the better answer than a frozen copy of the model does."""
    lead = (ours_better - frozen_better) - (ours_worse - frozen_worse)
    return -logsigmoid(beta * lead).mean()
```

Each argument is a row of log chances, one for each pair in the batch, made by lesson 27's `answer_log_chances`. The two `frozen` rows come from the reference and carry no slopes.

## The proof

```bash
uv run pytest tests/test_prefer.py
```

- With our model equal to the reference the loss is ln 2. It is lower when the better story has gained 2 nats on the reference, and higher when it has lost 2.
- The hand example gives 0.513.
- One step on a pair makes the better answer gain on the worse one, against the copy made before the step.

The pairs are lesson 28's, made from the 1,600 stories of lesson 27, with the same 60 orders held back. Training is 200 steps of 8 pairs at a rate of 0.00001. As in lesson 28, the readers also score the model's new stories before and after.

```bash
uv run python -m gpt.prefer
```

```
pairs to learn from: 1,386, pairs held back: 240. A story is 178 tokens on average
lesson 25's model: 30 of 200 tasks passed, 95% interval 0.10 to 0.21
lesson 25's model: 0.57 of the asked words used, 95% interval 0.53 to 0.61
lesson 25's model: 196 stories end by themselves, plain stories 1.549 nats a token
step 1: 0.693 on 8 pairs. ln 2 is 0.693
step 50: 0.640 on 8 pairs. ln 2 is 0.693
step 100: 0.557 on 8 pairs. ln 2 is 0.693
step 150: 0.303 on 8 pairs. ln 2 is 0.693
step 200: 0.270 on 8 pairs. ln 2 is 0.693
held-back pairs where the better story gained on the worse one: 150 of 240, 95% interval 0.55 to 0.70
after the pairs: 35 of 200 tasks passed, 95% interval 0.13 to 0.23
after the pairs: 0.60 of the asked words used, 95% interval 0.56 to 0.63
after the pairs: 197 stories end by themselves, plain stories 1.556 nats a token
plain stories, after minus before: +0.007 nats, 95% interval +0.006 to +0.008
the readers give its stories 0.62 on average, where they gave 0.53. The change has a 95% interval of +0.04 to +0.13
```

The first loss is ln 2. By step 200 it is 0.270 on a batch of 8 pairs.

That fall is on pairs it trains on: by then the model has drawn 1,600 pairs from 1,386, and each story sits in about two of them, so most stories in a late batch have been pushed before. The fair test is the 240 pairs held back, from 60 orders it never trained on. There the better story gained on the worse one in 150, and the interval, 0.55 to 0.70, is clear of a half.

The stories it writes changed too. On 200 unseen orders the readers give 0.62 where they gave 0.53, and the change has an interval of +0.04 to +0.13, clear of zero. Lesson 28's draw from the same model gave 0.57.

197 of the 200 stories end, and 35 pass where 30 did, and the two intervals overlap.

Plain stories cost 0.007 nats. Lesson 25's cost 0.069, from lesson 21's model, in ten times the steps at a hundred times this rate.

## What it still cannot do

We trained the model in lessons 25 to 29 on what to write, never on what to refuse.

## Check yourself

Predict before you run: what is the loss on the first step, before the model has moved?

<details><summary>Answer</summary>

ln 2 = 0.693. The model starts as the reference, so each lead is 0 and each bet is even.
</details>

<details><summary>Our model makes both stories of a pair 5 nats more likely than the reference does. What is the lead?</summary>

0. Both gains are 5, and the lead is their difference. The loss rewards the better story for gaining on the worse one, and nothing else.
</details>

<details><summary>Lesson 25's loss keeps pulling on a story however likely it is. What does this loss do with a pair already far apart?</summary>

Little. Its loss is nearly flat at a large lead, so the push fades. Lesson 25's loss pulls until a story's chance is 1, which it never is.
</details>

<details><summary>From lesson 14: the readers' number rose from 0.53 to 0.62, with an interval clear of zero. What does that interval leave out?</summary>

The seed. It is drawn over the 200 orders, for one training run. Lesson 14 would run more seeds before calling a setting such as beta well chosen.
</details>

Go deeper:

- Rafailov and others, [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (2023). The loss on this page, and the proof that it is lesson 28's setting, held near a reference, with the scorer folded in.
- Lambert, [the RLHF book](https://rlhfbook.com/). It sets DPO beside the methods of lessons 27 and 28.
- Hugging Face, [TRL](https://github.com/huggingface/trl). A library with this loss and its relatives, ready to run.
