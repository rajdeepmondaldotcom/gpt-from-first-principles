"""Lesson 32. Batching: many requests share each pass of the model."""

from dataclasses import dataclass, field

import torch

from gpt.cost import cache_bytes
from gpt.generate import choose
from gpt.rope import rope


@dataclass
class Request:
    prompt: list  # token ids
    limit: int  # the most tokens to write
    stop: int | None = None
    how: dict = field(default_factory=dict)  # temperature, top_p, generator
    written: list = field(default_factory=list)
    kept: list | None = None  # its own cache, while it reads its prompt
    read: int = 0  # prompt tokens read so far

    def done(self):
        return bool(self.written) and (
            len(self.written) >= self.limit or self.written[-1] == self.stop
        )


def slide(kept, by, pad):
    """Put `pad` empty places before a cache and turn its keys `by` places on.

    Only the distance between two tokens enters a score (lesson 15), so a whole row can sit
    anywhere along the batch as long as its keys are turned to match.
    """
    for one in kept:
        k, v = one["k"], one["v"]
        places = torch.full((k.size(2),), by, device=k.device)
        empty = k.new_zeros(k.size(0), k.size(1), pad, k.size(3))
        one["k"], one["v"] = (
            torch.cat([empty, rope(k, places)], dim=2),
            torch.cat([empty, v], dim=2),
        )


class Engine:
    def __init__(self, model, budget, chunk=64, store=None):
        self.model, self.budget, self.chunk, self.store = model, budget, chunk, store
        self.reading, self.rows = [], []  # requests reading their prompt, and requests writing
        self.kept, self.real = None, None  # the batch's cache, and which of its places are real

    def bytes_for(self, request):
        return cache_bytes(self.model, len(request.prompt) + request.limit)

    def admit(self, request):
        """Take a request if the cache it may need still fits in the budget."""
        promised = sum(self.bytes_for(r) for r in self.reading + self.rows)
        if promised + self.bytes_for(request) > self.budget:
            return False
        request.kept = [{} for _ in self.model.blocks]
        if self.store:  # lesson 33: start from the stored beginning of this prompt, if there is one
            request.kept, request.read = self.store.fetch(request.prompt, len(self.model.blocks))
        self.reading.append(request)
        return True

    @torch.no_grad()
    def step(self):
        """One round of work. Returns each (request, token) written in it."""
        device, out = self.model.embed.weight.device, []
        if self.rows:  # one pass writes one token for every row
            last = torch.tensor([[r.written[-1]] for r in self.rows], device=device)
            self.real = torch.cat([self.real, self.real.new_ones(len(self.rows), 1)], dim=1)
            for one in self.kept:
                one["real"] = self.real
            scores = self.model(last, self.kept)[:, -1]
            for request, row in zip(self.rows, scores):
                request.written.append(choose(row, **request.how))
                out.append((request, request.written[-1]))
            self.leave()
        for request in list(self.reading):  # each new request reads one chunk of its prompt
            part = request.prompt[request.read : request.read + self.chunk]
            scores = self.model(torch.tensor([part], device=device), request.kept)[0, -1]
            request.read += len(part)
            if request.read == len(request.prompt):
                if self.store:
                    self.store.keep(request.prompt, request.kept)
                request.written.append(choose(scores, **request.how))
                out.append((request, request.written[-1]))
                self.reading.remove(request)
                if not request.done():
                    self.join(request)
        return out

    def join(self, request):
        """Add a request that has read its prompt to the batch, lined up at the batch's end."""
        mine = torch.ones(1, request.read, dtype=torch.bool, device=request.kept[0]["k"].device)
        if not self.rows:
            self.kept, self.real = request.kept, mine
        else:
            gap = self.real.size(1) - request.read  # how much longer the batch is than this row
            if gap > 0:
                slide(request.kept, gap, gap)
                mine = torch.cat([mine.new_zeros(1, gap), mine], dim=1)
            elif gap < 0:
                slide(self.kept, -gap, -gap)
                self.real = torch.cat([self.real.new_zeros(len(self.rows), -gap), self.real], dim=1)
            for ours, theirs in zip(self.kept, request.kept):
                ours["k"] = torch.cat([ours["k"], theirs["k"]])
                ours["v"] = torch.cat([ours["v"], theirs["v"]])
            self.real = torch.cat([self.real, mine])
        request.kept = None
        self.rows.append(request)

    def leave(self):
        """Drop the rows that have finished, and any padding that every row left has."""
        stay = [i for i, r in enumerate(self.rows) if not r.done()]
        if len(stay) == len(self.rows):
            return  # nobody finished: the batch stays as it is
        self.rows = [self.rows[i] for i in stay]
        if not stay:
            self.kept = self.real = None
            return
        self.real = self.real[stay]
        spare = int(self.real.any(dim=0).nonzero()[0])  # places that are padding in every row
        self.real = self.real[:, spare:]
        for one in self.kept:
            one["k"], one["v"] = one["k"][stay][:, :, spare:], one["v"][stay][:, :, spare:]
        if spare:
            slide(self.kept, -spare, 0)


if __name__ == "__main__":
    import json
    from pathlib import Path

    from gpt.data import read
    from gpt.generate import generate
    from gpt.measure import arrivals, report, serve
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

    # Eight requests that arrive two steps apart, on a clock that counts steps. Each must write
    # what it writes alone.
    engine, steps = Engine(model, float("inf")), 0
    asked = [Request(prompt, 32, how={"temperature": 0}) for prompt in prompts(8, 48)]
    line = [(2.0 * i, request) for i, request in enumerate(asked)]

    def counted(step=engine.step):
        global steps
        steps += 1
        return step()

    engine.step = counted
    serve(engine, line, clock=lambda: steps)
    same = sum(
        generate(model, r.prompt, 32, temperature=0)[len(r.prompt) :] == r.written for r in asked
    )
    say(f"8 requests that join a running batch two steps apart: {same} write what they write alone")
    long = Request(prompts(1, 192)[0], 32, how={"temperature": 0})
    serve(Engine(model, float("inf"), chunk=64), [(0.0, long)])
    whole = generate(model, long.prompt, 32, temperature=0)[192:] == long.written
    say(f"a prompt of 192 tokens read in chunks of 64: the same answer, {whole}")

    # Throughput against the number of requests in one batch.
    sizes = {}
    for rows in (1, 2, 4, 8, 16, 32):
        tries = []
        for _ in range(4):  # the first try warms up; of the other three, keep the fastest
            line = [(0.0, Request(prompt, 64, how=how)) for prompt in prompts(rows)]
            tries.append(report(serve(Engine(model, float("inf"), chunk=256), line)))
        numbers = sizes[rows] = max(tries[1:], key=lambda one: one["tokens_a_second"])
        say(
            f"{rows} at once: {numbers['tokens_a_second']:,.0f} tokens a second in all, "
            f"{1 / numbers['seconds_a_token_p50']:,.0f} for each reader"
        )

    # Lesson 31's three rates of arrival again, with room for 16 requests at once.
    one = Engine(model, 0).bytes_for(Request(prompts(1)[0], 64))
    capacity = json.loads(Path("results/31_measure.json").read_text())["capacity"]
    say(f"the cache for one request of 64 + 64 tokens: {one:,} bytes")
    loads = {}
    for share in (0.5, 0.9, 1.5):
        chance = torch.Generator().manual_seed(1)  # lesson 31's arrivals, to the moment
        line = arrivals(prompts(120), share * capacity, chance, **how)
        numbers = loads[share] = report(serve(Engine(model, 16 * one, chunk=256), line))
        say(
            f"{share * capacity:.1f} requests a second, room for 16: first token after "
            f"{numbers['first_token_p50'] * 1e3:,.0f} ms for half, "
            f"{numbers['first_token_p95'] * 1e3:,.0f} ms for 95 in 100, "
            f"{numbers['tokens_a_second']:,.0f} tokens a second in all"
        )
    taken = Engine(model, one - 1).admit(Request(prompts(1)[0], 64))
    say(f"a request offered one byte less than its cache needs is taken: {taken}")
    save(
        "32_engine",
        torch=torch.__version__,
        device="cpu",
        same=same,
        sizes={str(k): v for k, v in sizes.items()},
        loads={str(k): v for k, v in loads.items()},
        bytes_for_one=one,
    )
