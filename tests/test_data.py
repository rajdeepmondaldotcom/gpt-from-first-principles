import unicodedata
from pathlib import Path

import pytest
import torch

from gpt.data import clean, held_out, minhash, near_repeats, overlap, prepare, read, shingles, write
from gpt.gpt2 import batch

LINES = Path("data/tiny_shakespeare.txt").read_text().split("\n\n")
DOCS = [doc for doc in LINES if len(doc) > 400][:200]  # 200 different speeches


def test_clean_keeps_text_and_drops_junk():
    assert clean("  " + DOCS[0] + "\n\n") == DOCS[0]
    assert clean("One day, a boy named Tom") is None  # a fragment
    assert clean(DOCS[0][:200] + "�" + DOCS[0][200:]) is None  # broken bytes
    assert clean("0.31 0.44 0.92 | 17 22 | " * 20) is None  # mostly not words


def test_one_character_written_two_ways_becomes_one():
    two_marks = unicodedata.normalize("NFD", "café")  # e, then a separate accent: 5 characters
    assert len(two_marks) == 5 and two_marks != "café"
    assert clean(two_marks + " " + DOCS[0]) == clean("café " + DOCS[0])


def test_overlap_by_hand():
    assert overlap({1, 2, 3}, {2, 3, 4}) == 2 / 4
    assert shingles("a b c d e f", n=5) == shingles("A  b c d e f", n=5)  # two runs of five words
    assert len(shingles("a b c d e f", n=5)) == 2


def test_minhash_agrees_as_often_as_the_sets_overlap():
    generator = torch.Generator().manual_seed(0)
    a, b = (torch.randint(1, 2**31 - 1, (2000,), generator=generator) for _ in range(2))
    items = torch.randint(2**31, (150,), generator=generator).tolist()
    left, right = set(items[:100]), set(items[50:])  # 50 shared of 150: an overlap of 1/3
    agree = (minhash(left, a, b) == minhash(right, a, b)).float().mean().item()
    assert agree == pytest.approx(1 / 3, abs=0.03)


def test_every_planted_mess_is_removed_and_nothing_else():
    messy = list(DOCS)
    messy.insert(30, DOCS[3].upper())  # a repeat in capitals
    messy.insert(70, DOCS[10].replace(" ", "  "))  # a repeat with other spacing
    messy.insert(90, DOCS[20] + "\n")  # an exact repeat
    messy.insert(120, DOCS[40].replace(" the ", " a ", 1))  # near-repeats: one word changed
    messy.insert(150, DOCS[60] + " Exeunt.")  # ... and a few words added
    messy.insert(160, "lorem")  # junk
    messy.insert(180, "�" * 300)
    messy.insert(200, "12 34 56 78 | " * 40)
    kept, counts = prepare(messy)
    assert kept == DOCS
    assert counts == {"read": 208, "junk": 3, "repeats": 3, "near_repeats": 2, "kept": 200}


def test_near_repeats_keeps_the_first_and_spares_different_texts():
    assert near_repeats(DOCS) == {}
    assert near_repeats([DOCS[0], DOCS[1], DOCS[0] + " Exit."]) == {2: 0}


def test_a_document_is_in_one_split_and_stays_there():
    unseen = {doc for doc in DOCS if held_out(doc, one_in=10)}
    assert 5 < len(unseen) < 40  # about a tenth of 200
    shuffled = sorted(DOCS, key=len)
    assert {doc for doc in shuffled if held_out(doc, one_in=10)} == unseen  # order does not matter
    assert unseen.isdisjoint(doc for doc in DOCS if not held_out(doc, one_in=10))


def test_batches_from_disk_equal_batches_from_memory(tmp_path):
    ids = torch.randint(4097, (5000,), generator=torch.Generator().manual_seed(0)).tolist()
    write(tmp_path / "ids.bin", ids)
    on_disk, in_memory = read(tmp_path / "ids.bin"), torch.tensor(ids)
    assert (tmp_path / "ids.bin").stat().st_size == 4 * 5000
    for mine, theirs in zip(
        batch(on_disk, 16, 8, torch.Generator().manual_seed(1)),
        batch(in_memory, 16, 8, torch.Generator().manual_seed(1)),
    ):
        assert torch.equal(mine, theirs)
