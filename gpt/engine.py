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
        self.rows = [self.rows[i] for i in stay]
        if not stay:
            self.kept = self.real = None
            return
        self.real = self.real[stay]
        spare = int(self.real.any(dim=0).nonzero()[0])  # places that are padding in every row
        self.real = self.real[:, spare:]
        for one in self.kept:
            one["k"], one["v"] = one["k"][stay][:, :, spare:], one["v"][stay][:, :, spare:]
        slide(self.kept, -spare, 0)
