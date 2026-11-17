# 24. Evaluation

By the end of this page each score and each comparison of two models carries a 95% interval, and each task a pass rate.

We can score the model on unseen text and read its stories. We cannot yet say how sure a score is, or whether a second model is better than this one.

## The idea

Lesson 21 scored the model at 1.565 nats a token with lesson 10's `evaluate`: windows of 256 that sit side by side. A report on a score should answer four questions.

1. How much text each token had before it. In side-by-side windows the first token of each window is scored with nothing before it. Move the window on by a quarter of its length, a stride of 64, and score only the tokens not scored before. The same 100 stories score 1.533 nats at a stride of 256 and 1.439 at 64. Read one story at a time, they score 1.444.
2. Whether the model has seen the answers. We call an unseen story that is also in the training text a leak. Lesson 18 dropped near-repeats before it split, and here the finder compares each unseen story with each training story. We have also used the unseen text once, when lesson 21 picked its learning rate from three. A careful report keeps a third set that no choice has touched.
3. How sure the score is. It is a mean over the stories we happened to hold out, and other stories would give another mean. To see how far it could move, draw 2,525 stories from our 2,525, with repeats allowed, and take the mean again. Do that 1,000 times. The middle 95% of those means is a 95% interval, and drawing again like this is the bootstrap. Draw whole stories and not single tokens: the tokens of one story rise and fall together, so they are not separate pieces of evidence.
4. Whether one model is better than another. Score both on the same stories, take the difference for each token, and put an interval on the mean difference. If the interval holds 0, these stories have not separated the two models.

A loss measures prediction, and a task checks one job with a rule that says pass or fail. We have two tasks. The easy one opens a story with a name, and passes when the story uses the name again and reaches its end. The hard one asks, in three lines, for a story with a given name and three given words.

A pass rate is a mean of passes and failures, but the bootstrap serves it badly when passes are few. Draw again from 100 failures and each draw is 100 failures, so the interval has no width. So a pass rate gets the range worked out by hand below: each rate at which a count this low, and one this high, still comes 1 time in 40.

We also ask for the chance that at least one of k tries passes, called pass at k. Given n tries of which c passed, count the ways to pick k tries that all failed, divide by the ways to pick any k, and take the result from 1.

## By hand

Pass at 2, from 4 tries of which 2 passed. There are 6 ways to pick 2 tries from 4. One of them picks the two failures.

1 - 1/6 = 0.833

The plain formula, 1 - (1 - 2/4)² = 0.75, is 0.083 too low here, and too low on average over many cases.

No pass in 100 tries. At a pass rate of 0.036, 100 failures in a row have a chance of 0.964¹⁰⁰ = 0.026, about 1 in 40. A 95% interval leaves that share outside at each end, so ours runs from 0 to 0.036. One pass in 100 runs up to 0.054 by the same sum, taken over one pass or none.

## The code

```python
# gpt/evaluate.py
def losses(model, ids, context, stride):
    """The loss of each token after the first. Each is scored once, with up to `context` before it."""
    assert stride <= context, "a stride past the context would skip tokens"
    out = []
    for start in range(0, len(ids) - 1, stride):
        end = min(start + stride, len(ids) - 1)  # this pass scores tokens start + 1 to end
        window = ids[max(0, end - context) : end + 1]
        scores = model(window[None, :-1])[0, start - end :]  # only the places not scored before
        out.append(F.cross_entropy(scores, window[start - end :].long(), reduction="none"))
    return torch.cat(out)
```

`F` is `torch.nn.functional`. `F.cross_entropy` with `reduction="none"` is lesson 3's loss with one number for each token and no mean taken, and `.long()` gives it the ids as 64-bit whole numbers. `start - end` is negative, so `[start - end :]` keeps the last `end - start` places.

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

- Each token is scored once at a stride of 1, 5 or 16. At a stride equal to the context the losses are lesson 10's, and `losses` refuses a stride past the context.
- The finder finds three planted leaks, one in capitals and one with words added, and nothing else.
- On made-up data with a known mean, the interval drawn block by block holds it 88 to 99 times in 100, and drawn value by value under 60.
- 70 passes of 100 give 0.60 to 0.79, and no pass gives 0 to 0.036. For a task passed 1 time in 50, the range holds the true rate in at least 19 runs of 20, where the bootstrap's top end falls short in over 1 run in 10.
- Pass at k gives 0.833 for the hand example, and averages to the true value where the plain formula is more than 0.03 too low.
- A story passes its own task and fails its neighbour's. `Timber` does not count as `Tim`.

```bash
uv run python -m gpt.evaluate
```

```
unseen stories: 2,525, training stories: 253,479
unseen stories: 1.480 nats a token, 95% interval 1.463 to 1.497
250 stories, drawn story by story: 0.094 wide. Drawn token by token: 0.033 wide
100 stories, 21,498 tokens, each story read alone: 1.444 nats a token
the same tokens as one run of text, stride 256: 1.533 nats a token
the same tokens as one run of text, stride 64: 1.439 nats a token
the same tokens as one run of text, stride 16: 1.439 nats a token
500 stories, 16 bits minus 32 bits: +0.0001 nats, 95% interval -0.0000 to +0.0002
three unseen stories planted among 20,000 training ones: found [3, 40, 77]
the real split, each unseen story against each training story: 0 leaks
a name to carry on: 97 of 100, 95% interval 0.91 to 0.99
a name to carry on: pass at 1 0.97, pass at 5 1.00, from 10 tries on each of 20
a story to order: 0 of 100, 95% interval 0.00 to 0.04
a story to order: pass at 1 0.00, pass at 5 0.00, from 10 tries on each of 20
a name to carry on: 'Once upon a time, there was a little cat named Ollie.'
a story to order: 'Name: Ollie\nWords: fish, stone, duck\nStory:'
```

All 2,525 unseen stories, read one at a time, score 1.480 nats, against lesson 21's 1.565 in side-by-side windows. The first 250 give an interval 0.094 nats wide drawn story by story and 0.033 drawn token by token, a third as wide as it should be.

With lesson 12's 16-bit switch on while scoring, 500 stories score 0.0001 nats worse, and the interval reaches 0, so these stories have not separated the two. Given a name to carry on, the model passes 97 of 100.

## What it still cannot do

Asked in three lines for a name and three words, the model passes 0 of 100, and none of 10 tries on each of 20. It was trained to continue stories, and no story in its text begins with a line that says what to write.

## Check yourself

Predict before you run: a model passes 0 of 10 tries on a task. What does the formula give for pass at 5?

<details><summary>Answer</summary>

0. Each way of picking 5 tries picks 5 failures.
</details>

<details><summary>From lesson 14: what in that lesson played the part that stories play here?</summary>

Seeds. There each seed gave one difference between two models that had the same luck. Here each story gives its tokens' differences, and stories are what we draw again.
</details>

Go deeper:

- Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (2024). Intervals for model scores, with questions grouped as our tokens are by story.
- Chen and others, [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (2021). Where the formula for pass at k comes from.
- UK AI Security Institute, [Inspect](https://inspect.aisi.org.uk/). A tool for writing tasks with checks, as labs run them.
