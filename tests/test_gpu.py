from types import SimpleNamespace

import torch

import gpt.gpu
from gpt.data import held_out, read, write
from gpt.gpu import more, pick, race, step_seconds
from gpt.model import GPT
from gpt.scale import predict, shape, steps_for

SMALL = {"vocab": 65, "context": 16, "width": 32, "heads": 2, "layers": 2}


def test_a_step_time_is_the_fastest_after_the_first_and_the_16_bit_switch_is_on(monkeypatch):
    ticks = iter([0, 5, 7, 7, 10])  # the first step starts at 0. The next two take 2 s and 3 s
    monkeypatch.setattr(gpt.gpu, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    seen, real = [], gpt.gpu.loss_on

    def watched(*args):
        seen.append(torch.is_autocast_enabled("cpu"))
        return real(*args)

    monkeypatch.setattr(gpt.gpu, "loss_on", watched)
    ids = torch.randint(65, (2000,), generator=torch.Generator().manual_seed(0))
    assert step_seconds(GPT(**SMALL), ids, size=4, repeats=2, half=True) == 2
    assert seen == [True, True, True]
    ticks = iter([0, 5, 7, 7, 10])
    step_seconds(GPT(**SMALL), ids, size=4, repeats=2)
    assert seen[3:] == [False, False, False]


def test_more_text_keeps_unseen_documents_out_and_writes_each_kept_one(tmp_path, words):
    docs = [f"story number {i} " + "and then the cat sat down again " * 4 for i in range(300)]
    docs += [docs[0], "too short"]  # a repeat and a fragment
    counts = more(tmp_path / "all.bin", docs, words)
    ids = read(tmp_path / "all.bin")
    assert counts["read"] == 302 and counts["junk"] == 1 and counts["repeats"] == 1
    assert 0 < counts["written"] < counts["kept"]  # some documents belong to the unseen text
    assert (ids == words.end).sum() == counts["written"]  # one end token for each written
    kept = words.decode([i for i in ids.tolist() if i != words.end])
    unseen = [doc.strip() for doc in docs[:300] if held_out(doc.strip())]
    assert unseen and not any(doc in kept for doc in unseen)


def test_more_text_puts_the_lead_before_each_document(tmp_path, words):
    game = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7"
    docs = [f"{game} g{i}" for i in range(40)]
    more(tmp_path / "games.bin", docs, words, tidy=lambda doc: doc, lead=" ")
    assert words.decode(read(tmp_path / "games.bin")[:6].tolist()) == " e4 e5"


def test_the_pick_is_the_smallest_shape_the_law_cannot_tell_from_the_best():
    fitted = {"alpha": 0.25, "beta": 0.55, "parts": [0.1, 1.4, 0.8], "miss": 0.05}
    family = [shape(32, 2), shape(64, 2), shape(128, 4)]
    rows, chosen = pick(fitted, family, 1e13, most=float("inf"))
    best = min(row["predicted"] for row in rows)
    near = [row for row in rows if row["predicted"] - best <= 0.05]
    assert chosen == min(near, key=lambda row: row["parameters"])
    for row in rows:  # each row is the law's own sum for that size and that many tokens
        model = GPT(**row["shape"])
        assert row["steps"] == steps_for(model, 1e13) and row["tokens"] == row["steps"] * 32 * 256
        assert row["predicted"] == predict(fitted, row["parameters"], row["tokens"])
    # A law that cannot tell any two apart takes the smallest.
    assert pick(fitted | {"miss": 9}, family, 1e13, float("inf"))[1]["shape"] == family[0]
    # A shape that reads too much is out, and "the best" is then the best of those left.
    left = rows[1:]
    best = min(row["predicted"] for row in left)
    near = [row for row in left if row["predicted"] - best <= 0.05]
    capped = pick(fitted, family, 1e13, most=rows[1]["tokens"])[1]
    assert capped == min(near, key=lambda row: row["parameters"])
    assert rows[0]["predicted"] < best - 0.05  # so a best taken over each shape would fit none


def test_two_workers_that_share_slopes_end_where_one_worker_on_the_whole_batch_does(tmp_path):
    torch.manual_seed(0)
    write(tmp_path / "ids.bin", torch.randint(65, (4000,)).tolist())
    alone, single = race(1, SMALL, tmp_path / "ids.bin", 4, 8, "gloo", tmp_path / "one.pt")
    shared, pair = race(2, SMALL, tmp_path / "ids.bin", 4, 4, "gloo", tmp_path / "two.pt")
    assert alone > 0 and shared > 0
    start = GPT(**SMALL).state_dict()  # not the seed's weights: the workers moved from their own
    moved = max((single[name] - start[name]).abs().max().item() for name in single)
    apart = max((single[name] - pair[name]).abs().max().item() for name in single)
    assert moved > 1e-3 and apart < 1e-5
