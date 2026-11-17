"""Lesson 18. Data: from a raw download to tokens on disk that a model can train on."""

import hashlib
import json
import unicodedata
import urllib.request
import zlib
from pathlib import Path

import torch

from gpt.bpe import END, Tokenizer, train
from gpt.results import save, say

# 2.2 GB of short stories written for small children. We take the first 200 MB of one fixed copy.
URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-train.txt"
)
SHA256 = "ee5242d9d94d6df3b5813e01adb0ee278eb77aa3bb7395fbd22cbe6e368869e9"
PRIME = 2**31 - 1


def fetch(path, megabytes=200):
    """Download the first `megabytes` of the stories, once, and check they are the bytes we used."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        ask = urllib.request.Request(URL, headers={"Range": f"bytes=0-{megabytes * 2**20 - 1}"})
        with urllib.request.urlopen(ask) as reply:
            path.write_bytes(reply.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, f"{path} is not the same slice"


def clean(doc):
    """One document in, tidy text out. Returns None for junk."""
    doc = unicodedata.normalize("NFC", doc).strip()  # one way to write each character
    plain = sum(ch.isalpha() or ch.isspace() for ch in doc)
    if len(doc) < 100 or "�" in doc or plain < 0.8 * len(doc):
        return None  # a fragment, broken bytes, or mostly not words
    return doc


def digest(text):
    """A short fingerprint: the same text always gives the same 20 bytes."""
    return hashlib.sha1(text.encode()).digest()


def shingles(doc, n=5):
    """The set of each run of n words in a row, each run stored as a number."""
    words = doc.lower().split()
    return {
        zlib.crc32(" ".join(words[i : i + n]).encode()) for i in range(max(1, len(words) - n + 1))
    }


def overlap(a, b):
    """The share of two sets' items that are in both. 1 for equal sets, 0 for no item in common."""
    return len(a & b) / len(a | b)


def minhash(items, a, b):
    """Scramble the items in len(a) different ways. Keep the smallest result of each way."""
    x = torch.tensor(sorted(items)) % PRIME
    return ((a[:, None] * x + b[:, None]) % PRIME).min(dim=1).values


def near_repeats(docs, least=0.8, bands=8, rows=4):
    """Each document that shares at least `least` of its word runs with an earlier one.

    Returns a dictionary: the index of the later document, then the index of the one it repeats.
    """
    generator = torch.Generator().manual_seed(0)
    a, b = (torch.randint(1, PRIME, (bands * rows,), generator=generator) for _ in range(2))
    first, drop = {}, {}  # first[key]: the earliest kept document with that band of numbers
    for i, doc in enumerate(docs):
        mine = shingles(doc)
        keys = [
            hash((band, *row))
            for band, row in enumerate(minhash(mine, a, b).view(bands, rows).tolist())
        ]
        alike = {first[key] for key in keys if key in first}  # shares a whole band: worth a look
        match = next((j for j in alike if overlap(mine, shingles(docs[j])) >= least), None)
        if match is None:
            for key in keys:
                first.setdefault(key, i)
        else:
            drop[i] = match
    return drop


def prepare(docs):
    """Clean each document, then drop repeats and near-repeats. Returns the rest, and the counts."""
    cleaned = [clean(doc) for doc in docs]
    unique, seen = [], set()
    for doc in cleaned:
        mark = digest(" ".join(doc.lower().split())) if doc else None  # blind to case and spacing
        if doc and mark not in seen:
            seen.add(mark)
            unique.append(doc)
    drop = near_repeats(unique)
    kept = [doc for i, doc in enumerate(unique) if i not in drop]
    junk = cleaned.count(None)
    counts = {"read": len(docs), "junk": junk, "repeats": len(docs) - junk - len(unique)}
    return kept, counts | {"near_repeats": len(drop), "kept": len(kept)}


def held_out(doc, one_in=100):
    """Whether a document belongs to the unseen text. Its own fingerprint decides, so it never moves."""
    return int.from_bytes(digest(doc)[:4], "big") % one_in == 0


def write(path, ids):
    """Token ids to disk, 4 bytes each."""
    path.write_bytes(torch.tensor(ids, dtype=torch.int32).numpy().tobytes())


def read(path):
    """Token ids from disk. The file is mapped, not loaded: only the parts a batch touches are read."""
    return torch.from_file(str(path), dtype=torch.int32, size=path.stat().st_size // 4)


if __name__ == "__main__":
    cache = Path("data/cache")
    fetch(cache / "tinystories.txt")
    docs = (cache / "tinystories.txt").read_text().split(END)[:-1]  # the slice cuts the last story
    kept, counts = prepare(docs)
    say(f"documents read: {counts['read']:,}")
    say(f"junk: {counts['junk']:,}, repeats: {counts['repeats']:,}")
    say(f"near-repeats: {counts['near_repeats']:,}")
    say(f"documents kept: {counts['kept']:,}")

    unseen = [doc for doc in kept if held_out(doc)]
    seen = [doc for doc in kept if not held_out(doc)]
    sample = "\n".join(seen[:6000])  # about 5 MB: enough to find the common pairs
    vocab = train(sample, 4096)
    tokenizer = Tokenizer(vocab)
    (cache / "stories_vocab.json").write_text(json.dumps([token.hex() for token in vocab]))
    sizes = {}
    for name, part in [("train", seen), ("val", unseen)]:
        ids = [i for doc in part for i in (*tokenizer.encode(doc), tokenizer.end)]
        write(cache / f"stories_{name}.bin", ids)
        size = sum(len(doc.encode()) for doc in part)
        sizes[name] = {"documents": len(part), "bytes": size, "tokens": len(ids)}
        say(
            f"{name}: {len(part):,} documents, {size / 1e6:.1f} MB, {len(ids):,} tokens, "
            f"{size / len(ids):.2f} bytes per token"
        )
    longest = sorted(vocab, key=len)[-6:]
    say(f"longest tokens: {[token.decode() for token in longest]}")
    save("18_data", torch=torch.__version__, counts=counts, sizes=sizes, slice_sha256=SHA256)
