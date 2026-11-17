"""Lesson 31. Latency and throughput: what to measure when a model serves people."""

import math
import time

import torch

from gpt.engine import Request


def percentile(values, q):
    """The smallest value that at least q percent of the values are at or below."""
    ranked = sorted(values)
    return ranked[max(0, math.ceil(q / 100 * len(ranked)) - 1)]


def arrivals(prompts, rate, chance, limit=64, **how):
    """A request for each prompt. They come at random moments, `rate` of them a second on average.

    `chance` is the generator that draws the moments. `how` goes to each request as it is.
    """
    gaps = torch.empty(len(prompts)).exponential_(rate, generator=chance)
    return [
        (at, Request(prompt, limit, how=how))
        for at, prompt in zip(gaps.cumsum(0).tolist(), prompts)
    ]


def serve(engine, arrivals, clock=time.perf_counter):
    """Run an engine until its work is done. `arrivals` is a list of (time, request), in order.

    Returns one record for each request: when it arrived and when each of its tokens was written.
    A request with no room waits in line. One too big for an empty engine is refused: no tokens.
    """
    began, waiting = clock(), list(arrivals)
    records = {id(request): {"arrived": at, "tokens": []} for at, request in arrivals}
    while waiting or engine.reading or engine.rows:
        now = clock() - began
        while waiting and waiting[0][0] <= now:
            if not engine.admit(waiting[0][1]) and (engine.reading or engine.rows):
                break  # no room yet: it waits, and so does everyone behind it
            waiting.pop(0)
        if not (engine.reading or engine.rows):
            if waiting:  # nothing to do until the next arrival: move the clock to it
                began = clock() - waiting[0][0]
            continue
        for request, _ in engine.step():
            records[id(request)]["tokens"].append(clock() - began)
    return list(records.values())


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


if __name__ == "__main__":
    from pathlib import Path

    from gpt.data import read
    from gpt.engine import Engine
    from gpt.results import save, say
    from gpt.story import load

    cache = Path("data/cache")
    model, words = load(cache / "story")
    text = read(cache / "stories_val.bin")
    generator = torch.Generator().manual_seed(0)
    how = {"temperature": 0.8, "generator": generator}

    def prompts(count, length=64):
        """`count` prompts, each `length` tokens of unseen text."""
        return [text[300 * i : 300 * i + length].tolist() for i in range(count)]

    def run(budget, line):
        """Serve a line of arrivals with this much memory for caches. Returns the report."""
        engine = Engine(model, budget, chunk=256)
        serve(engine, [(0.0, Request(prompts(1)[0], 8, how=how))])  # a first call, to warm up
        return report(serve(engine, line))

    # One request alone: the prompt is read in one pass, then each token takes a pass of its own.
    engine = Engine(model, float("inf"), chunk=256)
    serve(engine, [(0.0, Request(*prompts(1, 192), 64, how=how))])  # a first call, to warm up
    alone = serve(engine, [(0.0, Request(*prompts(1, 192), 64, how=how))])
    first = alone[0]["tokens"][0]
    gap = (alone[0]["tokens"][-1] - first) / 63
    say(f"alone, a prompt of 192 tokens: first token after {first * 1e3:.0f} ms")
    say(f"alone, the prompt is read at {192 / first:,.0f} tokens a second")
    say(f"alone, the answer is written at {1 / gap:,.0f} tokens a second, {gap * 1e3:.1f} ms each")

    # Memory speed: the fastest of five copies of 256 MB.
    big, copies = torch.ones(2**26), []
    for _ in range(5):
        began = time.perf_counter()
        big.clone()
        copies.append(time.perf_counter() - began)
    speed = big.nbytes / min(copies)
    weights = 4 * sum(p.numel() for p in model.parameters())
    say(f"this machine copies {speed / 1e9:.1f} GB a second")
    say(
        f"a token reads {weights / 1e6:.1f} MB of weights: at that speed, "
        f"{speed / weights:,.0f} tokens a second"
    )

    # A service that takes one request at a time, at three rates of arrival.
    one = Engine(model, 0).bytes_for(Request(prompts(1)[0], 64))  # the cache one request needs
    service = 64 / (64 / (192 / first) + 63 * gap)  # tokens a second if it never rested
    capacity = service / 64  # requests a second
    say(f"one request at a time can finish {capacity:.1f} requests a second")
    loads = {}
    for share in (0.5, 0.9, 1.5):
        chance = torch.Generator().manual_seed(1)  # the same 120 arrivals each time, only faster
        line = arrivals(prompts(120), share * capacity, chance, **how)
        numbers = loads[share] = run(one, line)
        say(
            f"{share * capacity:.1f} requests a second: first token after "
            f"{numbers['first_token_p50'] * 1e3:,.0f} ms for half, "
            f"{numbers['first_token_p95'] * 1e3:,.0f} ms for 95 in 100, "
            f"{numbers['tokens_a_second']:,.0f} tokens a second in all"
        )
    save(
        "31_measure",
        torch=torch.__version__,
        device="cpu",
        alone={"first": first, "gap": gap},
        copy_bytes_a_second=speed,
        capacity=capacity,
        loads={str(k): v for k, v in loads.items()},
    )
