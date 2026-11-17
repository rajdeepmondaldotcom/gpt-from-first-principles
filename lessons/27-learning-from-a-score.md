# 27. Learning from a score

By the end of this page the model has been trained with no right answers, only a number for each story it wrote, and you have measured that against fine-tuning on the stories that pass.

Both fine-tunes copied 60,404 stories that obeyed their orders, and the better one still leaves out 0.44 of the asked words. The check sees each miss, and nothing learns from it.

## The idea

Lesson 24's check says whether a story has the name and the three words. A finer version counts them: the share of those four that the story has, from 0 to 1. Call that number the score. Lesson 25's share left the name out.

The first way needs nothing new: let the model write many stories, keep the ones that pass, and fine-tune on those as lesson 25 did. This throws away the stories that failed, though one a word short came close.

The second way uses them all. Write the average score the model would earn as a sum: each story it might write, times its chance, times its score. We want that sum's slope for each weight. We cannot list them all, so we rewrite the sum to sample it. By lesson 2's table, the slope of ln x is the slope of x over x, so the slope of a chance is the chance times the slope of its logarithm. Each term of the sum then starts with a chance, and the slope is an average, over stories the model itself wrote, of

score × slope of the log of the story's chance

The log of a story's chance is the sum of the logs of its tokens' chances, lesson 1's surprise with the sign turned. So one training step is: let the model write some stories, score each, and step so that each story's log chance rises in proportion to its score.

The step as written is noisy. With scores between 0 and 1, no story is pushed down, good or bad, and only the difference between pushes teaches anything. So we write four stories for each order and subtract the average of the four from each score. A story above its group's average is made more likely, and one below it less.

Subtracting a number that does not depend on which story was drawn leaves the average slope where it was. The average of four does depend on it. But a score minus the average of four is three quarters of the score minus the average of the other three, and those three do not depend on this story. So on average the step keeps its direction and is three quarters as long.

Both ways get the same budget: 1,600 stories written, 16 to a round, and 100 steps at a peak rate of 0.0001 on lesson 25's schedule.

## By hand

Two orders, two stories each. The first order's stories score 1 and 0. The second order's both score 1.

| Story | Score | Its group's average | Score minus average |
|---|---|---|---|
| First order, story 1 | 1 | 0.5 | +0.5 |
| First order, story 2 | 0 | 0.5 | -0.5 |
| Second order, story 1 | 1 | 1 | 0 |
| Second order, story 2 | 1 | 1 | 0 |

The step pushes the first story's log chance up and the second's down as hard. The second order teaches nothing this round: both its stories scored the same.

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

A story the model wrote is stored as one of lesson 25's examples, so `collate` and its marks for the answer's places serve again. `rewards` is the usual name for the scores, and `baseline` for what is subtracted from them. `groups` gives each story the number of its order.

## The proof

```bash
uv run pytest tests/test_reinforce.py
```

- An answer's log chance equals the sum, token by token, of the logs of the chances the model gave.
- A made-up model picks one of 5 answers, and 2 of them score 1. The average of 200,000 sampled slopes matches the exact slope of its average score to 0.005.
- With the average of all the scores subtracted, the sampled slopes still average to the exact one, and their variance, the square of the spread, falls by more than 40%.
- With the scores in groups of four, the sampled slopes average to three quarters of the exact one. Scaled back up, their variance is more than 20% below the plain rule's.
- The hand example gives -0.125, +0.125, 0 and 0, and one step on two answers scored 1 and 0 raises the first's log chance and lowers the other's.

```bash
uv run python -m gpt.reinforce
```

```
lesson 25's model: 30 of 200 tasks passed, 95% interval 0.10 to 0.21
lesson 25's model: 0.57 of the asked words used, 95% interval 0.53 to 0.61
lesson 25's model: 196 stories end by themselves, plain stories 1.549 nats a token
first try: 192 of 1600 stories pass, and we keep those
keep what passes: 37 of 200 tasks passed, 95% interval 0.13 to 0.25
keep what passes: 0.57 of the asked words used, 95% interval 0.53 to 0.61
keep what passes: 191 stories end by themselves, plain stories 1.612 nats a token
round 20: the last 320 stories scored 0.58 on average
round 40: the last 320 stories scored 0.67 on average
round 60: the last 320 stories scored 0.70 on average
round 80: the last 320 stories scored 0.75 on average
round 100: the last 320 stories scored 0.73 on average
learn from the score: 56 of 200 tasks passed, 95% interval 0.22 to 0.35
learn from the score: 0.68 of the asked words used, 95% interval 0.64 to 0.71
learn from the score: 141 stories end by themselves, plain stories 1.573 nats a token
asked words used, learning from the score minus keeping what passes: +0.10, 95% interval +0.07 to +0.14
```

Lesson 25's model passes 30 of 200 here and passed 27 there: the stories are drawn afresh. Keeping what passes had 192 stories to learn from and reaches 37. The intervals overlap and the share of asked words stays at 0.57: no measured gain.

Learning from the score passes 56. Its training stories scored 0.58 over the first 20 rounds and 0.73 over the last 20. It uses 0.68 of the asked words against 0.57, a difference of +0.105 with an interval of +0.07 to +0.14.

The score counts a name and three words, and gives nothing for an ending. Only 141 of its 200 stories reach their end, where 196 did. Plain stories score 1.573 nats where they scored 1.549, and 1.612 after keeping what passes.

## What it still cannot do

The score gave nothing for an ending, and 59 of 200 stories no longer reach one. A reader would mark that down, and can say which of two stories is better with no written rule.

## Check yourself

Predict before you run: which will use more of the asked words after 1,600 stories, the model that kept what passed or the one that learned from each score?

<details><summary>Answer</summary>

The one that learned from each score: 0.68 against 0.57. Keeping what passes learned from 192 stories, the score from all 1,600.
</details>

<details><summary>One story of four scores 1 and the other three score 0. Work out its score minus the group's average, and three quarters of its score minus the average of the other three.</summary>

Both are 0.75: 1 - 0.25, and 3/4 × (1 - 0).
</details>

<details><summary>From lesson 1: why is the log of a story's chance a sum over its tokens?</summary>

The chance of a story is the product of the chances of its tokens, each given the ones before it. The log of a product is the sum of the logs.
</details>

Go deeper:

- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (1992). The paper that named this rule REINFORCE, with a baseline.
- OpenAI, [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html). The same derivation, with baselines.
- Shao and others, [DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024). It introduces a version with a group of answers for each question.
