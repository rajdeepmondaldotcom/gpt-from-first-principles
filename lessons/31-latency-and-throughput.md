# 31. Latency and throughput

By the end of this page you can say how long a reader waits and how much the machine gets done, and you have watched both as readers arrive faster and faster.

Each script so far served one reader: one prompt in, one story out. A service has many readers, and they arrive when they like. Before we change the code, we need numbers that tell a change that helped from one that did not.

## The idea

A reader feels two waits. The first is the wait before anything appears, called the time to the first token. The model has to read the whole prompt before it can write one token, so this wait grows with the prompt. The second is the gap between one token and the next. With lesson 23's cache, each new token costs one pass over one token.

The person who pays for the machine watches a third number: the tokens written each second, added up over all readers. That is the throughput, and the two waits are the latency.

A mean hides the readers who waited longest, so we sort the waits and read off percentiles. The 50th percentile is the smallest wait that half the readers were at or under. The 95th is the wait that 1 reader in 20 went over.

Readers do not arrive evenly. The script draws each gap between two arrivals by chance, at a set rate on average, so some readers come in a bunch and some after a pause. A request that arrives while the machine is busy waits in line.

The code that takes requests in and writes their tokens is the engine. It refuses a request too big for the machine's memory. Here it takes one request at a time, and lesson 32 opens it up.

Long waits come from the line. Say the machine can finish 3 requests a second. At half that rate, a request seldom finds another ahead of it. Near that rate, a bunch of arrivals builds a line that takes a long time to drain. Above that rate, the line grows for as long as readers keep coming.

A ceiling on the speed of writing: one new token reads each weight once, as lesson 17 counted, and the story model's weights are 21.5 MB. The script times a copy of 268 MB. A copy reads each byte and writes it, so it moves 537 MB through memory. That speed divided by 21.5 MB is the most tokens a second memory alone would allow, and we set our speed beside it.

## By hand

Ten waits, in milliseconds: 5, 1, 9, 3, 7, 2, 8, 4, 6, 10. Sorted, they are 1 to 10. Half of ten is five, so the 50th percentile is the fifth wait: 5. For the 95th, 0.95 × 10 = 9.5, which rounds up to the tenth wait: 10.

A slow tail. Of 100 readers, 90 wait 0.1 s and 10 wait 3 s.

| Measure | Sum | Value |
|---|---|---|
| 50th percentile | the 50th wait of 100 | 0.1 s |
| 95th percentile | the 95th wait of 100 | 3.0 s |
| Mean | (90 × 0.1 + 10 × 3.0) / 100 | 0.39 s |

The mean is a wait that no reader had.

The most that one request at a time can finish. In the run below, a request with a prompt of 64 tokens and an answer of 64 takes 0.099 s with the machine to itself. With no rest between requests, that is 1 / 0.099 = 10.1 requests a second and no more.

## The code

```python
# gpt/measure.py
def percentile(values, q):
    """The smallest value that at least q percent of the values are at or below."""
    ranked = sorted(values)
    return ranked[max(0, math.ceil(q / 100 * len(ranked)) - 1)]
```

`math.ceil` rounds up, and the `- 1` is because Python counts places from 0.

```python
# gpt/measure.py
def report(records):
    """The numbers a person waiting and a person paying each care about."""
    served = [r for r in records if r["tokens"]]
    first = [r["tokens"][0] - r["arrived"] for r in served]  # the wait before anything appears
    gaps = [
        (r["tokens"][-1] - r["tokens"][0]) / (len(r["tokens"]) - 1)
        for r in served
        if len(r["tokens"]) > 1
    ]
    written = sum(len(r["tokens"]) for r in served)
    span = max(r["tokens"][-1] for r in served) - min(r["arrived"] for r in served)
    return {
        "served": len(served),
        "refused": len(records) - len(served),
        "first_token_p50": percentile(first, 50),
        "first_token_p95": percentile(first, 95),
        "seconds_a_token_p50": percentile(gaps, 50),
        "tokens_a_second": written / span,  # all users together
    }
```

A record holds the moment a request arrived and the moment each of its tokens was written. A function called `serve` makes the records: it feeds the engine each request at its arrival time, holds back in line the ones there is no room for yet, and notes the clock after each step.

## The proof

```bash
uv run pytest tests/test_measure.py
```

- The two hand examples give 5 and 10, and 0.1 s and 3.0 s.
- On a clock we control, which moves one second for each step of the engine, two requests get the first-token waits and the throughput worked out by hand.
- With room for one request, a second one that arrives at the same moment waits four steps and then writes what it would have written alone.
- A request too big for an empty engine is refused and gets no tokens.
- 2,000 arrivals drawn at 4 a second come 0.25 s apart on average, to within a tenth.
- An idle engine jumps to the next arrival and does not count the empty time as a wait.

```bash
uv run python -m gpt.measure
```

```
alone, a prompt of 192 tokens: first token after 8 ms
alone, the prompt is read at 24,647 tokens a second
alone, the answer is written at 539 tokens a second, 1.9 ms each
this machine moves 91.0 GB a second through memory
a token reads 21.5 MB of weights: at that speed, 4,231 tokens a second
alone, a prompt of 64 tokens and an answer of 64 take 0.10 s
one request at a time can finish 10.1 requests a second
5.0 requests a second: first token after 3 ms for half, 157 ms for 95 in 100, 275 tokens a second in all
9.1 requests a second: first token after 250 ms for half, 724 ms for 95 in 100, 489 tokens a second in all
15.1 requests a second: first token after 1,737 ms for half, 2,826 ms for 95 in 100, 631 tokens a second in all
```

Timings are from an M4 laptop, and they move from run to run.

Alone, a request waits 8 ms for its first token after a prompt of 192 tokens, then gets a token each 1.9 ms: 539 a second. Memory alone would allow 4,231, almost eight times as many. Lesson 37 looks for what holds it at 539.

At 5.0 requests a second, half the machine's 10.1, half the readers wait 3 ms for a first token and 1 in 20 waits over 157 ms. At 9.1 the middle wait is 250 ms, and at 15.1 it is 1,737 ms. For these shorter prompts the middle gap between tokens is 1.5 ms at all three rates: the line delays the first token, and the writing keeps its pace.

## What it still cannot do

The machine served one reader while the rest stood in line, and each pass of the model wrote one token for that one reader. A pass can carry many rows at once, as training has done since lesson 3.

## Check yourself

Predict before you run: a machine finishes 3 requests a second and 4.5 arrive each second. The line starts empty. After 20 seconds, about how many requests are in it?

<details><summary>Answer</summary>

90 arrive and 60 are finished, so about 30. The line grows by 1.5 a second for as long as that goes on.
</details>

<details><summary>A request has a prompt three times as long as another's. Which of its two waits is longer?</summary>

The wait for the first token: the model reads the whole prompt before it writes anything. The gap between tokens is one pass over one token for both.
</details>

<details><summary>From lesson 23: without the cache, what would happen to the gap between tokens as the answer grows?</summary>

It would grow with the answer. Each new token would mean reading the whole text again, prompt and all, so each gap would be longer than the one before.
</details>

Go deeper:

- Google, [Site Reliability Engineering](https://sre.google/sre-book/monitoring-distributed-systems/), chapter 6. Latency, traffic, errors and saturation: four things to watch on a service.
- kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). Where the limits on reading and writing speed come from.
- Harchol-Balter, [Performance Modeling and Design of Computer Systems](https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html). Why a line grows so fast near the rate a machine can serve.
