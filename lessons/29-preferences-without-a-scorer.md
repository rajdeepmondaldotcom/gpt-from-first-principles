# 29. Preferences without a scorer

By the end of this page the storyteller has learned from lesson 28's pairs directly, with no second model in between.

The scorer of lesson 28 is a second model to train, to keep and to be fooled. The pairs themselves already say which story is better, and we have not learned from them directly.

## The idea

A pair says that for one order, story A is better than story B. What should that do to the storyteller? It should make A more likely and B less likely. Our model gives each text a chance, so "more likely" has an exact meaning: the log of A's chance should rise and the log of B's should fall.

We cannot compare the two log chances as they stand. They differ for reasons that say nothing about which story is better: each token lowers a story's log chance, so length alone can put one story far below another. So we compare how far each has moved since training began. Keep a frozen copy of the model, made before this training starts, and call it the reference.

For each story, ask how much more likely our model makes it than the reference does: the log of its chance under our model, minus the log of its chance under the reference. Call that the story's gain. Then for a pair,

lead = gain of the better story - gain of the worse story

We want the lead to grow, and the loss is lesson 28's bet again, with the lead times a small number beta in the place of x. Three things follow from that one line.

1. At the start our model is the reference. Each gain is 0, each lead is 0, and the loss is ln 2.
2. The loss falls as the better story gains on the worse one. It does not care whether the better one rose or the worse one fell.
3. A pair with a large lead has little left to teach, because a sure bet cannot get much surer. That is what stops the push, and it keeps the model near the reference. Beta sets how large a lead counts as large. A lead is in nats, summed over a story, and our stories run to about 180 tokens. With our beta of 0.1, a lead of 10 nats, under 0.06 a token, makes x = 1 and a bet of 0.73.

This is called direct preference optimization, or DPO. It needs no scorer and no stories written during training. The pairs are fixed, and the reference's part of each lead can be worked out once.

Its authors show where the loss comes from. Take lesson 28's setting, and let a story's score be beta times its gain. The storyteller then is its own scorer, and the two models of lesson 28 fold into one.

## By hand

One pair. Log chances are sums over the story's tokens.

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

At the start, with our model equal to the reference, x is 0, the chance 0.5 and the loss 0.693.

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

The pairs are lesson 28's, made from the 1,600 stories of lesson 27, with the same 60 orders held back. The script also asks lesson 28's question: what do the readers give the model's new stories, before and after?

```bash
uv run python -m gpt.prefer
```

```
@@OUTPUT@@
```

@@RESULTS@@

## What it still cannot do

Each lesson since 25 has taught the model to do something better. None has taught it that there is something it must not do.

## Check yourself

Predict before you run: what is the loss on the first step, before the model has moved?

<details><summary>Answer</summary>

ln 2 = 0.693, or close to it. The model starts as the reference, so each lead is 0 and each bet is even.
</details>

<details><summary>Our model makes both stories of a pair 5 nats more likely than the reference does. What is the lead?</summary>

0. Both gains are 5, and the lead is their difference. The loss rewards the better story for gaining on the worse one, and nothing else.
</details>

<details><summary>What would go wrong with a loss that only raised the better story's log chance?</summary>

Nothing would stop it. The surest way to raise one story's chance is to give it all the chance there is, and the model would lose the rest of what it knows. Here a large lead earns almost nothing more, so the push fades.
</details>

<details><summary>From lesson 28: which of that lesson's two models has gone, and what does its job now?</summary>

The scorer has gone. A story's gain over the reference, times beta, stands in for its score, so the storyteller scores its own stories.
</details>

Go deeper:

- Rafailov and others, [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (2023). The loss on this page, and the proof that it is lesson 28 with the scorer folded in.
- Lambert, [the RLHF book](https://rlhfbook.com/). It sets DPO beside the methods of lessons 27 and 28.
- Hugging Face, [TRL](https://github.com/huggingface/trl). A library with this loss and its relatives, ready to run.
