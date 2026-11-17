# 27. Learning from a score

By the end of this page the model has been trained with no right answers at all, only a number for each story it wrote, and you have measured that against the simplest thing one could try.

Both fine-tunes learned from 60,404 stories that already obeyed their orders: a right answer to copy, for each example. For most things we want from a model, nobody has written the right answers. We have only a check that says pass or fail.

## The idea

We have a check and no examples. Lesson 24's check says whether a story has the name and the three words. A finer version counts them: the share of those four things that the story has, from 0 to 1. Call that number the score.

The first thing to try needs nothing new. Let the model write many stories. Keep the ones that pass. Fine-tune on those, as lesson 25 did. The model's own successes become its examples. This throws away each story that failed, though a failure says something too. And it treats a pass as a text to copy token by token, when only four of its words earned the pass.

The second way uses each story. Write the average score the model would earn as a sum: each story it might write, times its chance, times its score. We want the slope of that sum for each weight. No one can list each story, but we can draw some, and the sum has a property that makes a few draws enough. The slope of a chance is the chance times the slope of its logarithm. So the slope of the whole sum is an average, over stories the model itself wrote, of

score × slope of the log of the story's chance

The log of a story's chance is the sum of the logs of its tokens' chances, lesson 1's surprise with the sign turned. So one training step is: let the model write some stories, score each one, and take a step that raises the log chance of each story in proportion to its score. A story with a high score is made more likely. Nothing in the step needs a right answer.

Used as it stands, this is noisy. If each score is between 0 and 1, each story is pushed up, good and bad alike, and only the difference between pushes teaches anything. So we write four stories for each order and subtract the average of the four from each score. A story above its group's average is made more likely, and one below it less. Subtracting a number that does not depend on which story was drawn leaves the average slope where it was. A test checks both claims on a case small enough to work out exactly.

Both ways get the same budget here: 1,600 stories written, and 100 steps.

## By hand

Two orders, two stories each. The first order's stories score 1 and 0. The second order's both score 1.

| Story | Score | Its group's average | Score minus average |
|---|---|---|---|
| First order, story 1 | 1 | 0.5 | +0.5 |
| First order, story 2 | 0 | 0.5 | -0.5 |
| Second order, story 1 | 1 | 1 | 0 |
| Second order, story 2 | 1 | 1 | 0 |

The step raises the log chance of the first story and lowers the second's by the same amount. The second order teaches nothing this round: both stories did equally well, so there is no reason to prefer one.

The loss is minus the average of (score minus average) × log chance. Its slope for each of the four log chances is that story's number, with the sign turned, over 4: -0.125, +0.125, 0 and 0.

## The code

```python
# gpt/reinforce.py
def answer_log_chances(model, examples, pad):
    """For each example, the log of the chance the model gives its whole answer."""
    x, y, scored = collate(examples, pad, model.embed.weight.device)
    scores = model(x)
    each = -torch.stack(
        [cross_entropy(scores[row][scored[row]], y[row][scored[row]]) for row in range(len(x))]
    )
    return each * scored.sum(dim=1)  # cross_entropy gave a mean; the answer's log chance is a sum


def score_loss(log_chances, rewards, groups):
    """Raise the chance of answers that scored above their group's average, and lower the rest."""
    rewards = torch.as_tensor(rewards, dtype=log_chances.dtype, device=log_chances.device)
    groups = torch.as_tensor(groups, device=log_chances.device)
    baseline = torch.stack([rewards[groups == g].mean() for g in groups])
    return -((rewards - baseline) * log_chances).mean()
```

A story the model wrote is stored as one of lesson 25's examples, the prompt and then the story, so `collate` and its marks for the answer's places serve again. `rewards` is the usual name for the scores, and `baseline` for what is subtracted from them.

## The proof

```bash
uv run pytest tests/test_reinforce.py
```

- An answer's log chance equals the sum, token by token, of the logs of the chances the model gave.
- A made-up model picks one of 5 answers, and 2 of them score 1. The exact slope of its average score is worked out directly. The average of 200,000 sampled slopes matches it to 0.005.
- With the average score subtracted, the sampled slopes still average to the exact one, and their variance, the square of the spread, falls by more than 40%.
- The hand example gives -0.125, +0.125, 0 and 0.
- One step on a pair of answers, one scored 1 and one 0, raises the first one's log chance and lowers the other's.
- A story the model writes comes back as an example that starts with the prompt, and is scored on the story alone.

The script writes 3,200 stories on the processor and takes about an hour.

```bash
uv run python -m gpt.reinforce
```

```
@@OUTPUT@@
```

@@RESULTS@@

## What it still cannot do

The check here was a program: four words, found or not found. Most of what people want from a story has no such program. People can still say which of two stories they like better.

## Check yourself

Predict before you run: which will use more of the asked words after 1,600 stories, the model that kept what passed or the model that learned from each score?

<details><summary>Answer</summary>

@@PREDICT@@
</details>

<details><summary>All four stories for an order score 0.25. What does that order teach in this round?</summary>

Nothing. Each score equals the group's average, so each story's number is 0 and the step does not move for them.
</details>

<details><summary>Why subtract the group's average, when the plain rule already has the right slope on average?</summary>

The plain rule pushes each story up by its score, and what it learns is hidden in the differences between large pushes. Subtracting the average leaves only the differences, so the same number of stories gives a steadier step.
</details>

<details><summary>From lesson 1: why is the log of a story's chance a sum over its tokens?</summary>

The chance of a story is the product of the chances of its tokens, each given the ones before it. The log of a product is the sum of the logs.
</details>

Go deeper:

- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (1992). The rule on this page, where it was first written down.
- OpenAI, [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html). The same derivation, slowly, with baselines.
- Shao and others, [DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024). It introduces a version with a group of answers for each question, now a common way to train models on checks.
