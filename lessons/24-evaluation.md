# 24. Evaluation

By the end of this page a score comes with a range, a comparison of two models comes with a verdict, and a task has a pass rate.

We can score the model on unseen text and read its stories. We cannot yet say how sure a score is, or whether a second model is better than this one.

## The idea

A score answers four questions, whether or not anyone asked them.

How much did each token get to read? Lesson 10 scored unseen text in windows that sit side by side, so the first token of each window was scored with nothing before it. Move the window on by a quarter of its length each time, score only the tokens not scored before, and each token has at least 192 tokens behind it. The distance moved is called the stride. The same model scores 1.592 nats at a stride of 256 and 1.499 at 64. One story at a time, from its first token, it scores 1.480. A score means little until we know how the text was cut.

Has the model seen the answers? An unseen story that is also in the training text is called a leak, and it makes the score look better than it is. Lesson 18 split by fingerprint so that this cannot happen, and here we check: its near-repeat finder runs across the two sets.

How sure is the score? It is a mean over the stories we happened to hold out, and other stories would give another mean. To see how far it could move, draw 2,525 stories at random from our 2,525, with repeats allowed, and work out the mean again. Do that 1,000 times. The range that holds the middle 95% of those means is called a 95% interval, and the method is called the bootstrap.

Draw whole stories and not single tokens. The tokens of one story rise and fall together: a hard story is hard from start to end. A thousand tokens from five stories are closer to five pieces of evidence than to a thousand.

Is one model better than another? Score both on the same stories and take the difference for each token, as lesson 14 took it for each seed. Then put an interval on the mean difference. If the interval holds 0, the stories do not tell the two models apart.

A loss says how well a model predicts. A task says whether it can do a job. Ours: write a story that uses a given name and three given words, checked exactly. For a task with some luck in it we also ask for the chance that at least one of k tries passes, called pass at k. Given n tries of which c passed, count the ways to pick k tries that all failed, and divide by the ways to pick any k.

## By hand

Pass at 2, from 4 tries of which 2 passed. There are 6 ways to pick 2 tries from 4. One of them picks the two failures.

1 - 1/6 = 0.833

The plain sum, 1 - (1 - 2/4)² = 0.75, reads low. A test shows that it reads low on average too.

## The code

```python
# gpt/evaluate.py
def losses(model, ids, context, stride):
    """The loss of each token after the first. Each is scored once, with up to `context` before it."""
    out = []
    for start in range(0, len(ids) - 1, stride):
        end = min(start + stride, len(ids) - 1)  # this pass scores tokens start + 1 to end
        window = ids[max(0, end - context) : end + 1]
        scores = model(window[None, :-1])[0, start - end :]  # only the places not scored before
        out.append(F.cross_entropy(scores, window[start - end :].long(), reduction="none"))
    return torch.cat(out)
```

`F.cross_entropy` with `reduction="none"` is lesson 3's loss with one number for each token and no mean taken.

```python
# gpt/evaluate.py
def interval(blocks, generator, draws=1000):
    """A 95% interval for the mean of every value. Whole blocks are drawn, never single values."""
    sums = torch.tensor([float(sum(block)) for block in blocks])
    counts = torch.tensor([float(len(block)) for block in blocks])
    drawn = torch.randint(len(blocks), (draws, len(blocks)), generator=generator)
    means = sums[drawn].sum(dim=1) / counts[drawn].sum(dim=1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()
```

A block is one story's losses. `drawn` holds 1,000 rows, each a random pick of stories. `quantile(0.025)` is the value that 2.5% of the means fall below.

## The proof

```bash
uv run pytest tests/test_evaluate.py tests/test_tasks.py
```

- Each token is scored once at a stride of 1, 5 or 16. At a stride of 1 each has as much before it as fits. At a stride equal to the context the losses are lesson 10's.
- Three planted leaks are found, one of them in capitals and one with words added, and nothing else is.
- On made-up data whose true mean we know, the interval drawn block by block holds it between 88 and 99 times in 100. Drawn value by value it holds it under 60 times in 100.
- 70 passes of 100 give an interval of 0.61 to 0.79.
- Pass at k gives 0.833 for the hand example, and averages to the true value where the plain sum reads low by more than 0.03.
- A story passes its own task and fails its neighbour's. `Timber` does not count as `Tim`.

```bash
uv run python -m gpt.evaluate
```

```
unseen documents: 2,525, and 24,350 of the training ones
20,000 tokens at a stride of 256: 1.592 nats a token
20,000 tokens at a stride of 64: 1.499 nats a token
20,000 tokens at a stride of 16: 1.500 nats a token
unseen stories: 1.480 nats a token, 95% interval 1.463 to 1.497
250 stories, drawn story by story: 0.094 wide. Drawn token by token: 0.033 wide
16 bits minus 32 bits: +0.0001 nats, 95% interval -0.0000 to +0.0002
three unseen stories planted in the training text: found [3, 40, 77]
the real split: 0 leaks
tasks passed: 0 of 100, 95% interval 0.00 to 0.00
pass at 1, from 10 tries on each of 20 tasks: 0.00
pass at 5, from 10 tries on each of 20 tasks: 0.00
a task: 'Name: Ollie\nWords: fish, stone, duck\nStory:'
```

The same 250 stories give an interval of 0.094 nats drawn story by story, and 0.033 drawn token by token. The second is a third as wide as it should be.

The run in 16 bits differs by 0.0001 nats, and its interval reaches 0. The stories do not tell the two apart.

## What it still cannot do

The model passes 0 of 100 tasks, and none of 10 tries on each of 20 more. It was trained to continue stories, and no story in its text begins with a line that says what to write.

## Check yourself

Predict before you run: a model passes 0 of 10 tries on a task. What does the formula give for pass at 5?

<details><summary>Answer</summary>

0. Each way of picking 5 tries picks 5 failures.
</details>

<details><summary>Two models score 1.480 and 1.470 nats on the same stories. What do you need before calling the second one better?</summary>

The difference for each token on the same stories, and an interval on its mean, drawn story by story. If the interval holds 0, you have not shown it.
</details>

<details><summary>Why is 1.592 at a stride of 256 not wrong?</summary>

It answers another question: how well the model predicts when a window can start anywhere. It is the right score to set beside training, which cuts the text the same way.
</details>

<details><summary>From lesson 14: what in that lesson played the part that stories play here?</summary>

Seeds. There the differences were taken seed by seed, and here token by token within the same stories. In both, the two models meet the same luck.
</details>

Go deeper:

- Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (2024). Intervals for model scores, with questions grouped as our tokens are grouped by story.
- Chen and others, [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (2021). Where the formula for pass at k comes from.
- UK AI Security Institute, [Inspect](https://inspect.aisi.org.uk/). A framework for tasks with checks, as labs run them.
