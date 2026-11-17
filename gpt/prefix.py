"""Lesson 33. Prefix caching: read the beginning that many prompts share once, and keep it."""

from collections import OrderedDict

import torch


class PrefixStore:
    """Pieces of cache, each for one block of tokens, found again by the text up to that block."""

    def __init__(self, budget, block=16):
        self.budget, self.block, self.used = budget, block, 0
        self.pieces = (
            OrderedDict()
        )  # the tokens up to a block's end -> that block's keys and values

    def fetch(self, ids, blocks):
        """A cache holding the longest stored beginning of `ids`, and how many tokens it covers."""
        found, end = [], self.block
        while end < len(ids) and tuple(ids[:end]) in self.pieces:  # leave a token to read
            self.pieces.move_to_end(tuple(ids[:end]))  # used now, so the last to be thrown out
            found.append(self.pieces[tuple(ids[:end])])
            end += self.block
        if not found:
            return [{} for _ in range(blocks)], 0
        kept = [
            {name: torch.cat([piece[i][name] for piece in found], dim=2) for name in ("k", "v")}
            for i in range(blocks)
        ]
        return kept, end - self.block

    def keep(self, ids, kept):
        """Store each whole block of a prompt's cache that is not stored yet."""
        for end in range(self.block, len(ids) + 1, self.block):
            if tuple(ids[:end]) not in self.pieces:
                piece = [
                    {name: one[name][:, :, end - self.block : end].clone() for name in ("k", "v")}
                    for one in kept
                ]
                self.pieces[tuple(ids[:end])] = piece
                self.used += sum(t.nbytes for one in piece for t in one.values())
        while self.used > self.budget:  # full: throw out the piece used longest ago
            _, piece = self.pieces.popitem(last=False)
            self.used -= sum(t.nbytes for one in piece for t in one.values())
