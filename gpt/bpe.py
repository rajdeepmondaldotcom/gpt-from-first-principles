"""Lesson 5. Tokens: cut text into pieces bigger than a character."""

import math
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

from gpt.results import save, say

# Cut text into chunks before merging, so no token spans two words. A chunk is a run of
# letters, a run of digits or a run of anything else, each with the one space before it.
SPLIT = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+")
END = "<|endoftext|>"  # marks where one document stops and the next begins


def merge(tokens, pair):
    """Join each occurrence of `pair` into one token, reading left to right."""
    out, i = [], 0
    while i < len(tokens):
        if tuple(tokens[i : i + 2]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return tuple(out)


def train(text, size):
    """Byte pair encoding. Returns the vocabulary: the bytes of each token mapped to its id."""
    vocab = {bytes([b]): b for b in range(256)}  # start with one token per byte
    chunks = Counter(SPLIT.findall(text))  # each distinct chunk, and how often it occurs
    chunks = {tuple(bytes([b]) for b in c.encode()): n for c, n in chunks.items()}
    # shortcut: recounts every pair after each merge; fine for a few thousand merges on a few MB.
    while len(vocab) < size:
        pairs = Counter()
        for tokens, n in chunks.items():
            for pair in pairwise(tokens):
                pairs[pair] += n
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))  # most frequent; ties to the smallest
        vocab[best[0] + best[1]] = len(vocab)
        chunks = {merge(tokens, best): n for tokens, n in chunks.items()}
    return vocab


class Tokenizer:
    def __init__(self, vocab, split=SPLIT):
        self.vocab = vocab
        self.split = split
        self.end = len(vocab)  # the id of END, one past the last ordinary token
        self.bytes = {i: token for token, i in vocab.items()} | {self.end: END.encode()}
        self.cache = {}

    def encode_chunk(self, chunk):
        """Start from bytes. Keep joining the pair that was learned earliest."""
        # shortcut: rescans the chunk after each join, so a chunk of n bytes costs n squared;
        # fine for words, replace before encoding text from strangers.
        parts = [bytes([b]) for b in chunk.encode()]
        while len(parts) > 1:
            rank, i = min(
                (self.vocab.get(a + b, math.inf), i) for i, (a, b) in enumerate(pairwise(parts))
            )
            if rank == math.inf:
                break
            parts[i : i + 2] = [parts[i] + parts[i + 1]]
        return [self.vocab[p] for p in parts]

    def encode(self, text, special=False):
        """Text to token ids. END becomes one token only when `special` is set."""
        ids = []
        for part in text.split(END) if special else [text]:
            for chunk in self.split.findall(part):
                if chunk not in self.cache:
                    self.cache[chunk] = self.encode_chunk(chunk)
                ids += self.cache[chunk]
            ids.append(self.end)
        return ids[:-1]

    def decode(self, ids):
        return b"".join(self.bytes[i] for i in ids).decode(errors="replace")


if __name__ == "__main__":
    import torch

    from gpt.mlp_lm import MLP, evaluate
    from gpt.mlp_lm import train as train_mlp

    text = Path("data/tiny_shakespeare.txt").read_text()
    n = int(len(text) * 0.9)
    words = re.compile("[a-z]+")  # a word is a run of letters, whatever its case
    seen, later = set(words.findall(text[:n].lower())), words.findall(text[n:].lower())
    unseen = sum(word not in seen for word in later)
    say(f"words in the training text: {len(seen):,} different ones")
    say(f"words in the unseen text: {len(later):,}, of which {unseen:,} never occur in training")

    scores = {}
    for size in (512, 1024):
        tokenizer = Tokenizer(train(text[:n], size))
        ids, val_ids = (torch.tensor(tokenizer.encode(part)) for part in (text[:n], text[n:]))
        per_token = (len(text) - n) / len(val_ids)
        torch.manual_seed(0)
        model = MLP(size + 1)  # lesson 4's model, reading eight tokens in place of eight characters
        train_mlp(model, ids, steps=30000, rate=0.1, generator=torch.Generator().manual_seed(0))
        bits = evaluate(model, val_ids) / math.log(2) / per_token
        scores[size] = {"bytes_per_token": per_token, "val_bits_per_byte": bits}
        say(f"{size} tokens: {per_token:.2f} bytes per token, MLP {bits:.3f} bits per byte")

    learned = [tokenizer.bytes[i].decode() for i in range(256, 1024)]
    pieces = [tokenizer.decode([i]) for i in tokenizer.encode("Once upon a time")]
    say(f"first merges: {learned[:12]}")
    say(f"longest tokens: {sorted(learned, key=len)[-6:]}")
    say(f"'Once upon a time' -> {pieces}")
    save(
        "05_bpe",
        torch=torch.__version__,
        seed=0,
        words=len(seen),
        unseen_words=unseen,
        held_out_words=len(later),
        first_merges=learned[:12],
        scores=scores,
    )
