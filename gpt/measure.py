"""Lesson 31. Latency and throughput: what to measure when a model serves people."""

import math
import time


def percentile(values, q):
    """The smallest value that at least q percent of the values are at or below."""
    ranked = sorted(values)
    return ranked[max(0, math.ceil(q / 100 * len(ranked)) - 1)]


def serve(engine, arrivals, clock=time.perf_counter):
    """Run an engine until its work is done. `arrivals` is a list of (time, request), in order.

    Returns one record for each request: when it arrived and when each of its tokens was written.
    A request the engine refuses gets no tokens.
    """
    began, waiting = clock(), list(arrivals)
    records = {id(request): {"arrived": at, "tokens": []} for at, request in arrivals}
    while waiting or engine.reading or engine.rows:
        now = clock() - began
        while waiting and waiting[0][0] <= now:
            engine.admit(waiting.pop(0)[1])
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
