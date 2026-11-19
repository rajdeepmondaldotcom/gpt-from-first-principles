# 31. Latency and throughput

By the end of this page you can say how long a listener waits and how much the machine gets done, and you have watched both as listeners arrive faster and faster.

Every story so far came from a script, one order at a time. We have not measured how long someone asking for a story waits, or how much the machine writes in a second. A service has many such listeners, and they arrive when they like.

## The idea

A listener feels two waits. The first is the wait before anything appears, called the time to the first token. The model has to read the whole prompt before it can write one token, so this wait grows with the prompt. The second is the gap between one token and the next. With lesson 23's cache, each new token costs one pass over one token.

Whoever pays for the machine watches a third number: the tokens written each second, added up over all listeners. That is the throughput, and the two waits are the latency.

A mean hides the listeners who waited longest, so we sort the waits and read off percentiles. The 50th percentile is the smallest wait that half the listeners were at or under. The 95th is the smallest wait that 95 listeners in 100 were at or under.

Listeners do not arrive evenly. The script draws each gap between two arrivals by chance, at a set rate on average, so some listeners come in a bunch and some after a pause. A request that arrives while the machine is busy waits in line.

The code that takes requests in and writes their tokens is the engine. It turns away a request too big for the machine's memory. It takes one request at a time until we widen it in lesson 32. The script times lesson 21's story model, which has the storyteller's shape.

Long waits come from the line. Say the machine can finish 3 requests a second. At half that rate, a request seldom finds another ahead of it. Near that rate, a bunch of arrivals builds a line that takes a long time to drain. Above that rate, the line grows as long as listeners keep coming.

A ceiling on writing speed: each new token reads each weight once (lesson 17), and the story model's weights are 21.5 MB. The script times a copy of 268 MB, which reads each byte and writes it: 537 MB through memory. Those bytes a second divided by 21.5 MB is the most tokens a second memory alone allows.

## By hand

Ten waits, in milliseconds: 5, 1, 9, 3, 7, 2, 8, 4, 6, 10. Sorted, they are 1 to 10. Half of ten is five, so the 50th percentile is the fifth wait: 5. For the 95th, 0.95 × 10 = 9.5, which rounds up to the tenth wait: 10.

A slow tail. Of 100 listeners, 90 wait 0.1 s and 10 wait 3 s.

| Measure | Sum | Value |
|---|---|---|
| 50th percentile | the 50th wait of 100 | 0.1 s |
| 95th percentile | the 95th wait of 100 | 3.0 s |
| Mean | (90 × 0.1 + 10 × 3.0) / 100 | 0.39 s |

No listener had a wait of 0.39 s.

The most that one request at a time can finish. In the run below, a request with a prompt of 64 tokens and an answer of 64 takes 0.099 s alone. With no rest between requests, that is 1 / 0.099 = 10.1 requests a second.

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
        "tokens_a_second": written / span,  # all listeners together
    }
```

A record holds the moment a request arrived and the moment each of its tokens was written. `serve` makes the records: it feeds the engine each request at its arrival time, holds back the ones there is no room for yet, and notes the clock after each step.

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

Alone, a 192-token prompt has a time to first token of 8 ms, read at 24,647 tokens a second: all 192 tokens pass through each weight in one read. Writing reads every weight for each token: 539 tokens a second. Memory alone would allow 4,231, almost eight times that. Lesson 37 looks for what holds it there.

The three rates are half, nine tenths and one and a half times the 10.1 above. After these prompts of 64 tokens the middle gap between tokens is 1.5 ms at all three rates: the line delays the first token, to 1,737 ms for half the listeners at 15.1 requests a second, and the writing keeps its pace. Each rate's 120 arrival times are one draw, and it came slow: 275 tokens a second is 4.3 requests of 64 tokens, not 5.

## What it still cannot do

The machine served one listener while the rest stood in line, and each pass of the model wrote one token for that one listener. A pass can carry many rows at once, as training has done since lesson 3.

## Check yourself

Predict before you run: a machine finishes 3 requests a second and 4.5 arrive each second. The line starts empty. After 20 seconds, about how many requests are in it?

<details><summary>Answer</summary>

90 arrive and 60 are finished, so about 30. The line grows by 1.5 a second for as long as that goes on.
</details>

<details><summary>A request has a prompt three times as long as another's. Which of its two waits grows more?</summary>

The wait for the first token. The gap grows less: 1.9 ms after a prompt of 192 tokens, 1.5 ms after 64, as each token reads a longer cache.
</details>

<details><summary>From lesson 23: without the cache, what would happen to the gap between tokens as the answer grows?</summary>

It would grow with the answer. Each new token would mean reading the whole text again, prompt and all, so each gap would be longer than the one before.
</details>

Go deeper:

- Google, [Site Reliability Engineering](https://sre.google/sre-book/monitoring-distributed-systems/), chapter 6. Latency, traffic, errors and saturation: four things to watch on a service.
- kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). Where the limits on reading and writing speed come from.
- Harchol-Balter, [Performance Modeling and Design of Computer Systems](https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html). Why a line grows so fast near the rate a machine can serve.
