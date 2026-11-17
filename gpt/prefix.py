"""Lesson 33. Prefix caching: read the beginning that many prompts share once, and keep it."""

from collections import OrderedDict

import torch


class PrefixStore:
    """Pieces of cache, each for a run of `piece` tokens, found again by the text up to its end."""

    def __init__(self, budget, piece=16):
        self.budget, self.piece, self.used = budget, piece, 0
        self.pieces = OrderedDict()  # the tokens up to a piece's end -> its keys and values

    def fetch(self, ids, blocks):
        """A cache holding the longest stored beginning of `ids`, and how many tokens it covers."""
        found, end = [], self.piece
        while end < len(ids) and tuple(ids[:end]) in self.pieces:  # leave a token to read
            found.append(self.pieces[tuple(ids[:end])])
            end += self.piece
        self.touch(ids, end - self.piece)
        if not found:
            return [{} for _ in range(blocks)], 0
        kept = [
            {name: torch.cat([piece[i][name] for piece in found], dim=2) for name in ("k", "v")}
            for i in range(blocks)
        ]
        return kept, end - self.piece

    def touch(self, ids, upto):
        """Mark a beginning's pieces as used now, the longest first. The shortest then goes last,
        so a piece is never thrown out while a longer one that needs it stays."""
        for end in range(upto, 0, -self.piece):
            self.pieces.move_to_end(tuple(ids[:end]))

    def keep(self, ids, kept):
        """Store each whole piece of a prompt's cache that is not stored yet."""
        for end in range(self.piece, len(ids) + 1, self.piece):
            if tuple(ids[:end]) not in self.pieces:
                piece = [
                    {name: one[name][:, :, end - self.piece : end].clone() for name in ("k", "v")}
                    for one in kept
                ]
                self.pieces[tuple(ids[:end])] = piece
                self.used += sum(t.nbytes for one in piece for t in one.values())
        self.touch(ids, len(ids) - len(ids) % self.piece)
        while self.used > self.budget:  # full: throw out the piece used longest ago
            _, piece = self.pieces.popitem(last=False)
            self.used -= sum(t.nbytes for one in piece for t in one.values())


if __name__ == "__main__":
    from pathlib import Path

    from gpt.cost import cache_bytes
    from gpt.data import read
    from gpt.engine import Engine, Request
    from gpt.measure import percentile, serve
    from gpt.results import save, say
    from gpt.story import load

    cache = Path("data/cache")
    model, words = load(cache / "story")
    text = read(cache / "stories_val.bin")
    shared = text[:160].tolist()  # the beginning that each prompt has
    ends = [text[300 * i : 300 * i + 16].tolist() for i in range(1, 21)]  # the part that changes
    counted = []
    model.embed.register_forward_hook(lambda part, inputs, out: counted.append(inputs[0].numel()))

    def run(prompts, store):
        """Serve the prompts one after another. Returns tokens read, each wait, and each answer."""
        engine = Engine(model, float("inf"), chunk=256, store=store)
        serve(engine, [(0.0, Request(prompts[0][-8:], 4, how={"temperature": 0}))])  # warm up
        counted.clear()
        waits, answers = [], []
        for prompt in prompts:
            request = Request(prompt, 32, how={"temperature": 0})
            waits.append(serve(engine, [(0.0, request)])[0]["tokens"][0])
            answers.append(request.written)
        return sum(counted), percentile(waits, 50), answers

    def show(name, tokens, wait):
        say(f"{name}: {tokens:,} tokens read, first token after {wait * 1e3:.0f} ms for half")

    last = [shared + end for end in ends]
    plain, wait, answers = run(last, None)
    show("20 prompts that share their first 160 tokens, no store", plain, wait)
    stored, wait, again = run(last, PrefixStore(budget=float("inf")))
    show("the same prompts with a store", stored, wait)
    say(f"the answers are the same with and without the store: {again == answers}")
    first, wait, _ = run([end + shared for end in ends], PrefixStore(budget=float("inf")))
    show("the part that changes put first, with a store", first, wait)
    room = 12 * cache_bytes(model, 16)  # twelve pieces of 16 tokens
    small = PrefixStore(budget=room)
    tight, wait, _ = run(last, small)
    show("a store with room for 12 pieces", tight, wait)
    say(f"that store holds {small.used:,} bytes of its {room:,}, in {len(small.pieces)} pieces")
    save(
        "33_prefix",
        torch=torch.__version__,
        device="cpu",
        tokens_read={"plain": plain, "stored": stored, "changing_first": first, "tight": tight},
    )
